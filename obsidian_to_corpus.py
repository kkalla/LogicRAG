#!/usr/bin/env python3
"""
Obsidian 마크다인 파일들을 LogicRAG corpus 형식으로 변환

특징:
1. 헤딩(#, ##, ###) 단위로 섹션 분할
2. 긴 섹션은 LLM으로 논리적 단위로 재분할

사용법:
    python obsidian_to_corpus.py --input /path/to/obsidian --output corpus.json

환경변수:
    OPENAI_API_KEY: LLM 분할용 (선택, 미설정 시 길이만으로 분할)
"""

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_PASSAGE_LENGTH = 2000
MIN_PASSAGE_LENGTH = 100


@dataclass
class Passage:
    title: str
    text: str
    level: int = 1
    source_file: str = ""


class ObsidianCorpusBuilder:
    """Obsidian vault에서 corpus를构建하는 클래스"""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        if not self.vault_path.exists():
            raise ValueError(f"Vault 경로가 존재하지 않습니다: {vault_path}")

    def find_markdown_files(self) -> List[Path]:
        """모든 마크다운 파일 찾기"""
        md_files = list(self.vault_path.rglob("*.md"))
        print(f"찾은 마크다운 파일: {len(md_files)}개")
        return md_files

    def parse_markdown_sections(self, content: str, file_title: str) -> List[Passage]:
        """마크다운에서 섹션 단위로 분할

        Returns:
            List[Passage]: 각 섹션을 Passage 객체로 반환
        """
        passages = []

        # 헤딩 패턴: #, ##, ### 등
        heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

        # 헤딩 위치 찾기
        matches = list(heading_pattern.finditer(content))

        if not matches:
            # 헤딩이 없으면 전체를 하나의 패시지로
            if content.strip():
                passages.append(
                    Passage(
                        title=file_title,
                        text=content.strip(),
                        level=1,
                        source_file=file_title,
                    )
                )
            return passages

        # 각 섹션 추출
        for i, match in enumerate(matches):
            level = len(match.group(1))  # # 개수
            heading_text = match.group(2).strip()

            # 섹션 시작 위치
            start = match.end()

            # 다음 섹션 시작 위치 (또는 문서 끝)
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(content)

            # 섹션 내용 추출
            section_content = content[start:end].strip()

            # 섹션 제목构建
            if level == 1:
                title = f"{file_title} - {heading_text}"
            else:
                title = f"{file_title} - {heading_text}"

            if section_content:
                passages.append(
                    Passage(
                        title=title,
                        text=section_content,
                        level=level,
                        source_file=file_title,
                    )
                )

        # 첫 번째 헤딩 이전의 내용 처리
        if matches[0].start() > 0:
            preamble = content[: matches[0].start()].strip()
            if preamble:
                passages.insert(
                    0,
                    Passage(
                        title=f"{file_title} - 개요",
                        text=preamble,
                        level=1,
                        source_file=file_title,
                    ),
                )

        return passages

    def split_long_passage_with_llm(self, passage: Passage) -> List[Passage]:
        """LLM으로 긴 패시지를 논리적 단위로 분할"""
        text = passage.text

        if len(text) <= MAX_PASSAGE_LENGTH:
            return [passage]

        # LLM이 없으면 문단 단위로 분할
        if not OPENAI_API_KEY:
            return self._split_by_paragraphs(passage)

        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)

            # 텍스트가 너무 길면 앞부분만 처리
            text_to_process = text[:5000] if len(text) > 5000 else text

            prompt = f"""다음 텍스트를 논리적으로 완결된 여러 단위로 나누어 주세요.
각 단위는 독립적으로 이해될 수 있어야 하며, 500-1000자 정도가 적당합니다.

원본 제목: {passage.title}

텍스트:
{text_to_process}

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
                        source_file=passage.source_file,
                    )
                    for unit in units
                ]

        except Exception as e:
            print(f"  LLM 분할 실패, 문단 단위로 대체: {e}")

        return self._split_by_paragraphs(passage)

    def _split_by_paragraphs(self, passage: Passage) -> List[Passage]:
        """문단 단위로 분할 (폴백)"""
        text = passage.text

        # 빈 줄을 기준으로 문단 분할
        paragraphs = re.split(r"\n\s*\n", text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) <= MAX_PASSAGE_LENGTH:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        if len(chunks) <= 1:
            return [passage]

        return [
            Passage(
                title=f"{passage.title} ({i+1}/{len(chunks)})",
                text=chunk,
                level=passage.level,
                source_file=passage.source_file,
            )
            for i, chunk in enumerate(chunks)
        ]

    def clean_text(self, text: str) -> str:
        """마크다운 텍스트 정리"""
        # 코드 블록은 유지
        # 이미지/링크 정리
        text = re.sub(r"!\[\[(.*?)\]\]", r"[이미지: \1]", text)  # Obsidian 이미지
        text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)  # Obsidian 링크
        text = re.sub(r"\[\[(.*?)\|(.*?)\]\]", r"\2", text)  # Obsidian 별칭 링크

        # 과도한 공백 정리
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def build_corpus(self, use_llm_split: bool = True) -> List[dict]:
        """전체 vault에서 corpus构建"""
        md_files = self.find_markdown_files()
        corpus = []
        total_passages = 0

        for md_file in md_files:
            # .obsidian 등 제외
            if any(part.startswith(".") for part in md_file.parts):
                continue

            relative_path = md_file.relative_to(self.vault_path)
            file_title = relative_path.stem

            print(f"  처리 중: {relative_path}")

            try:
                content = md_file.read_text(encoding="utf-8")
                content = self.clean_text(content)

                # 섹션 단위 분할
                passages = self.parse_markdown_sections(content, file_title)

                # 긴 패시지 분할
                for passage in passages:
                    if use_llm_split and len(passage.text) > MAX_PASSAGE_LENGTH:
                        split_passages = self.split_long_passage_with_llm(passage)
                    else:
                        split_passages = [passage]

                    for sp in split_passages:
                        if len(sp.text) >= MIN_PASSAGE_LENGTH:
                            corpus.append({"title": sp.title, "text": sp.text})
                            total_passages += 1

            except Exception as e:
                print(f"    오류: {e}")

        print(f"\n총 {total_passages}개 패시지 생성")
        return corpus


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Obsidian → LogicRAG Corpus")
    parser.add_argument("--input", type=str, required=True, help="Obsidian vault 경로")
    parser.add_argument(
        "--output", type=str, default="obsidian_corpus.json", help="출력 파일"
    )
    parser.add_argument("--no-llm", action="store_true", help="LLM 분할 사용 안 함")

    args = parser.parse_args()

    print("=" * 50)
    print("Obsidian → LogicRAG Corpus 변환")
    print("=" * 50)
    print(f"Vault: {args.input}")
    print(f"출력: {args.output}")

    builder = ObsidianCorpusBuilder(args.input)

    print(f"\n1. 마크다운 파일 스캔 중...")
    corpus = builder.build_corpus(use_llm_split=not args.no_llm)

    # 저장
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\n2. 저장 완료: {args.output}")

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
        print(f"내용:\n{corpus[0]['text'][:300]}...")


if __name__ == "__main__":
    main()
