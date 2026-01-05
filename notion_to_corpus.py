#!/usr/bin/env python3
"""
Notion 페이지들을 LogicRAG corpus 형식으로 변환하는 스크립트

특징:
1. 섹션/헤딩 단위로 분할
2. 긴 섹션은 LLM으로 논리적 단위로 재분할

사용법:
    python notion_to_corpus.py --output corpus.json

환경변수:
    NOTION_API_KEY: Notion Integration 키 (필수)
    OPENAI_API_KEY: LLM 분할용 (선택, 미설정 시 길이만으로 분할)
"""

import json
import os
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import requests

load_dotenv()

# 설정
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_PASSAGE_LENGTH = 2000  # 이 길이를 넘으면 LLM로 분할
MIN_PASSAGE_LENGTH = 100  # 이것보다 짧으면 병합 고려


@dataclass
class Passage:
    """하나의 패시지 단위"""

    title: str
    text: str
    level: int = 1  # 헤딩 레벨 (1=H1, 2=H2, 3=H3)
    source_page_id: str = ""


def extract_text_from_rich_text(rich_text: List[Dict]) -> str:
    """rich_text에서 순수 텍스트 추출"""
    if not rich_text:
        return ""

    texts = []
    for text_obj in rich_text:
        if "text" in text_obj:
            content = text_obj["text"].get("content", "")
            texts.append(content)
        elif text_obj.get("type") == "mention":
            mention_type = text_obj.get("mention", {}).get("type", "")
            if mention_type == "page":
                texts.append("[연결된 페이지]")
            elif mention_type == "database":
                texts.append("[데이터베이스]")

    return "".join(texts)


class NotionCorpusBuilder:
    """Notion 페이지에서 corpus를构建하는 클래스"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or NOTION_API_KEY
        if not self.api_key:
            raise ValueError("NOTION_API_KEY 환경 변수가 필요합니다")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Notion-Version": "2025-09-03",
        }

    def search_pages(
        self, query: str = None, database_id: str = None, limit: int = None
    ) -> List[Dict]:
        """Notion 페이지 검색"""
        search_url = "https://api.notion.com/v1/search"

        search_payload = {
            "filter": {"property": "object", "value": "page"},
            "page_size": 100 if limit is None else min(limit, 100),
        }

        if query:
            search_payload["query"] = query

        pages = []
        next_cursor = None

        while True:
            if next_cursor:
                search_payload["start_cursor"] = next_cursor

            response = requests.post(
                search_url, json=search_payload, headers=self.headers
            )
            response.raise_for_status()
            data = response.json()

            batch_pages = data.get("results", [])

            # database_id 필터링
            if database_id:
                batch_pages = [
                    p
                    for p in batch_pages
                    if p.get("parent", {}).get("database_id") == database_id
                ]

            pages.extend(batch_pages)

            if not data.get("has_more"):
                break
            if limit and len(pages) >= limit:
                break
            next_cursor = data.get("next_cursor")

        return pages[:limit] if limit else pages

    def get_page_title(self, page: Dict) -> str:
        """페이지 제목 추출"""
        properties = page.get("properties", {})

        for key, value in properties.items():
            if value.get("type") == "title":
                title_text = value.get("title", [])
                return extract_text_from_rich_text(title_text) or "Untitled"

        return "Untitled"

    def fetch_page_blocks(self, page_id: str) -> List[Dict]:
        """페이지의 모든 블록 가져오기 (재귀)"""
        all_blocks = []
        blocks_to_process = [page_id]

        while blocks_to_process:
            current_id = blocks_to_process.pop(0)

            url = f"https://api.notion.com/v1/blocks/{current_id}/children"
            response = requests.get(
                url, params={"page_size": 100}, headers=self.headers
            )
            response.raise_for_status()
            data = response.json()

            blocks = data.get("results", [])
            all_blocks.extend(blocks)

            # 자식이 있는 블록은 나중에 처리
            for block in reversed(blocks):
                if block.get("has_children"):
                    blocks_to_process.insert(0, block.get("id"))

        return all_blocks

    def blocks_to_passages(
        self, blocks: List[Dict], page_title: str, page_id: str
    ) -> List[Passage]:
        """블록들을 섹션 단위 패시지로 변환"""
        passages = []

        # 현재 섹션 상태
        current_section = {"h1": "", "h2": "", "h3": ""}
        current_content = []
        current_level = 1

        for block in blocks:
            block_type = block.get("type", "")

            # 헤딩 처리
            if block_type in ["heading_1", "heading_2", "heading_3"]:
                # 이전 섹션 저장
                if current_content:
                    title = self._build_section_title(
                        current_section, current_level, page_title
                    )
                    passages.append(
                        Passage(
                            title=title,
                            text="\n".join(current_content).strip(),
                            level=current_level,
                            source_page_id=page_id,
                        )
                    )
                    current_content = []

                # 새 섹션 시작
                level = int(block_type.split("_")[1])
                rich_text = block.get(block_type, {}).get("rich_text", [])
                heading_text = extract_text_from_rich_text(rich_text)

                if level == 1:
                    current_section["h1"] = heading_text
                    current_section["h2"] = ""
                    current_section["h3"] = ""
                elif level == 2:
                    current_section["h2"] = heading_text
                    current_section["h3"] = ""
                elif level == 3:
                    current_section["h3"] = heading_text

                current_level = level

            # 콘텐츠 블록 처리
            elif block_type == "paragraph":
                rich_text = block.get("paragraph", {}).get("rich_text", [])
                text = extract_text_from_rich_text(rich_text)
                if text:
                    current_content.append(text)

            elif block_type == "bulleted_list_item":
                rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                text = extract_text_from_rich_text(rich_text)
                if text:
                    current_content.append(f"• {text}")

            elif block_type == "numbered_list_item":
                rich_text = block.get("numbered_list_item", {}).get("rich_text", [])
                text = extract_text_from_rich_text(rich_text)
                if text:
                    # 번호는 나중에 처리
                    current_content.append(text)

            elif block_type == "to_do":
                rich_text = block.get("to_do", {}).get("rich_text", [])
                text = extract_text_from_rich_text(rich_text)
                checked = block.get("to_do", {}).get("checked", False)
                if text:
                    prefix = "[완료]" if checked else "[ ]"
                    current_content.append(f"{prefix} {text}")

            elif block_type == "code":
                rich_text = block.get("code", {}).get("rich_text", [])
                text = extract_text_from_rich_text(rich_text)
                if text:
                    current_content.append(f"```\n{text}\n```")

            elif block_type == "quote":
                rich_text = block.get("quote", {}).get("rich_text", [])
                text = extract_text_from_rich_text(rich_text)
                if text:
                    current_content.append(f"> {text}")

            elif block_type == "callout":
                rich_text = block.get("callout", {}).get("rich_text", [])
                text = extract_text_from_rich_text(rich_text)
                if text:
                    current_content.append(f"[참고] {text}")

            elif block_type == "divider":
                current_content.append("---")

            # 토글, 표 등은 간단히 처리
            elif block_type == "toggle":
                rich_text = block.get("toggle", {}).get("rich_text", [])
                text = extract_text_from_rich_text(rich_text)
                if text:
                    current_content.append(f"> {text}")

        # 마지막 섹션 저장
        if current_content:
            title = self._build_section_title(
                current_section, current_level, page_title
            )
            passages.append(
                Passage(
                    title=title,
                    text="\n".join(current_content).strip(),
                    level=current_level,
                    source_page_id=page_id,
                )
            )

        return passages

    def _build_section_title(self, section: Dict, level: int, page_title: str) -> str:
        """섹션 제목构建"""
        parts = []
        if section.get("h1"):
            parts.append(section["h1"])
        if section.get("h2"):
            parts.append(section["h2"])
        if section.get("h3"):
            parts.append(section["h3"])

        if not parts:
            return page_title

        return f"{page_title} - {' / '.join(parts)}"

    def split_long_passage_with_llm(self, passage: Passage) -> List[Passage]:
        """LLM으로 긴 패시지를 논리적 단위로 분할"""
        if not OPENAI_API_KEY:
            # LLM 없으면 문장 단위로 분할
            return self._split_by_sentences(passage)

        text = passage.text
        if len(text) <= MAX_PASSAGE_LENGTH:
            return [passage]

        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)

            prompt = f"""다음 텍스트를 논리적으로 완결된 여러 단위로 나누어 주세요.
각 단위는 독립적으로 이해될 수 있어야 하며, 500-1000자 정도가 적당합니다.

원본 제목: {passage.title}

텍스트:
{text[:4000]}  {'' if len(text) <= 4000 else '...(텍스트가 잘림, 뒷부분은 문장 단위로 처리)'}

JSON 형식으로 응답해주세요 (다른 텍스트 없이 JSON만):
{{
    "units": [
        {{"title": "단위 제목1", "text": "내용1"}},
        {{"title": "단위 제목2", "text": "내용2"}}
    ]
}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=3000,
            )

            result = json.loads(response.choices[0].message.content)
            units = result.get("units", [])

            if units:
                return [
                    Passage(
                        title=(
                            f"{passage.title} - {unit['title']}"
                            if unit.get("title")
                            else passage.title
                        ),
                        text=unit["text"],
                        level=passage.level,
                        source_page_id=passage.source_page_id,
                    )
                    for unit in units
                ]

        except Exception as e:
            print(f"LLM 분할 실패, 문장 단위로 대체: {e}")

        return self._split_by_sentences(passage)

    def _split_by_sentences(self, passage: Passage) -> List[Passage]:
        """문장 단위로 분할 (폴백)"""
        import re

        text = passage.text
        if len(text) <= MAX_PASSAGE_LENGTH:
            return [passage]

        # 문장 분할 (간단한 정규식)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = ""
        chunk_num = 1

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= MAX_PASSAGE_LENGTH:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                chunk_num += 1

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [
            Passage(
                title=f"{passage.title} ({i+1}/{len(chunks)})",
                text=chunk,
                level=passage.level,
                source_page_id=passage.source_page_id,
            )
            for i, chunk in enumerate(chunks)
        ]

    def build_corpus(self, pages: List[Dict], use_llm_split: bool = True) -> List[Dict]:
        """페이지 리스트에서 corpus构建"""
        corpus = []
        total_passages = 0

        for page in pages:
            page_id = page.get("id")
            page_title = self.get_page_title(page)

            print(f"  처리 중: {page_title}")

            # 블록 가져오기
            blocks = self.fetch_page_blocks(page_id)

            # 섹션 단위 패시지 변환
            passages = self.blocks_to_passages(blocks, page_title, page_id)

            # 긴 패시지 분할
            if use_llm_split:
                for passage in passages:
                    split_passages = self.split_long_passage_with_llm(passage)
                    for sp in split_passages:
                        if len(sp.text) >= MIN_PASSAGE_LENGTH:
                            corpus.append({"title": sp.title, "text": sp.text})
                            total_passages += 1
            else:
                for passage in passages:
                    if len(passage.text) >= MIN_PASSAGE_LENGTH:
                        corpus.append({"title": passage.title, "text": passage.text})
                        total_passages += 1

        print(f"\n총 {total_passages}개 패시지 생성")
        return corpus


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Notion → LogicRAG Corpus")
    parser.add_argument("--database-id", type=str, help="특정 database ID")
    parser.add_argument("--query", type=str, help="검색어 필터")
    parser.add_argument("--limit", type=int, help="최대 페이지 수")
    parser.add_argument(
        "--output", type=str, default="notion_corpus.json", help="출력 파일"
    )
    parser.add_argument("--no-llm", action="store_true", help="LLM 분할 사용 안 함")

    args = parser.parse_args()

    print("=" * 50)
    print("Notion → LogicRAG Corpus 변환")
    print("=" * 50)

    builder = NotionCorpusBuilder()

    print(f"\n1. 페이지 검색 중...")
    pages = builder.search_pages(
        query=args.query, database_id=args.database_id, limit=args.limit
    )
    print(f"   찾은 페이지: {len(pages)}개")

    print(f"\n2. 패시지 추출 중...")
    corpus = builder.build_corpus(pages, use_llm_split=not args.no_llm)

    # 저장
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\n3. 저장 완료: {args.output}")

    # 통계
    if corpus:
        print(f"\n=== Corpus 통계 ===")
        print(f"총 패시지 수: {len(corpus)}")
        avg_length = sum(len(p["text"]) for p in corpus) / len(corpus)
        print(f"평균 길이: {avg_length:.0f}자")
        print(f"최대 길이: {max(len(p['text']) for p in corpus)}자")
        print(f"최소 길이: {min(len(p['text']) for p in corpus)}자")

        print(f"\n=== 미리보기 (첫 번째 패시지) ===")
        print(f"제목: {corpus[0]['title']}")
        print(f"내용: {corpus[0]['text'][:200]}...")


if __name__ == "__main__":
    main()
