#!/bin/bash

# Agentic-RAG evaluation script
# This script runs evaluations on all available datasets with configurable RAG model(s)

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration parameters
MAX_ROUNDS=5
TOP_K=5
EVAL_TOP_KS="5 10"
LIMIT=50 # Number of questions to evaluate per dataset
MODELS_TO_RUN_STR="vanilla agentic light" # Default model(s): vanilla, agentic, light (space-separated)
TARGET_DATASETS_STR="" # Default: all datasets. Space separated e.g. "hotpotqa musique 2wikimultihopqa"

# Output directory for results
RESULTS_DIR="result"
mkdir -p $RESULTS_DIR

# Show usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help               Show this help message"
    echo "  -m, --max-rounds NUM     Set maximum rounds (default: $MAX_ROUNDS)"
    echo "  -k, --top-k NUM          Set top-k contexts (default: $TOP_K)"
    echo "  -l, --limit NUM          Set number of questions per dataset (default: $LIMIT)"
    echo "  -r, --model MODELS       Set RAG model(s) (space-separated: vanilla, agentic, light; default: "$MODELS_TO_RUN_STR")"
    echo "  -d, --datasets SETS      Set target dataset(s) (space-separated: hotpotqa, musique, 2wikimultihopqa; default: all)"
    echo ""
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -m|--max-rounds)
            MAX_ROUNDS="$2"
            shift 2
            ;;
        -k|--top-k)
            TOP_K="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -r|--model)
            MODELS_TO_RUN_STR="$2"
            shift 2
            ;;
        -d|--datasets)
            TARGET_DATASETS_STR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo -e "${GREEN}Starting Agentic-RAG evaluations...${NC}"
echo -e "Models to run: $MODELS_TO_RUN_STR"
echo -e "Target datasets: ${TARGET_DATASETS_STR:-All}"
echo -e "Max rounds: $MAX_ROUNDS, Top-K: $TOP_K, Eval Top-Ks: $EVAL_TOP_KS, Questions per dataset: $LIMIT"
echo ""

# Function to check if a dataset should be run
should_run_dataset() {
    local dataset_key=$1
    if [[ -z "$TARGET_DATASETS_STR" ]]; then
        return 0 # True, run it if TARGET_DATASETS_STR is empty (run all)
    fi
    for ds_token in $TARGET_DATASETS_STR; do
        if [[ "$ds_token" == "$dataset_key" ]]; then
            return 0 # True, dataset_key found in the list
        fi
    done
    return 1 # False, dataset_key not found
}

# Function to run evaluation on a dataset
# MODEL variable will be set by the outer loop for each model iteration
run_evaluation() {
    local dataset=$1
    local corpus=$2
    local output_filename=$3 # e.g., "${MODEL}_hotpotqa_evaluation.json"
    local dataset_name_log=$(basename "$dataset" .json)
    
    echo -e "${BLUE}Evaluating ${dataset_name_log} with ${MODEL} model${NC}"
    echo "Dataset: $dataset"
    echo "Corpus: $corpus"
    echo "Output: $output_filename"
    
    # Splitting EVAL_TOP_KS into an array to pass as individual arguments
    read -ra EVAL_TOP_KS_ARRAY <<< "$EVAL_TOP_KS"
    eval_top_ks_args=""
    for k in "${EVAL_TOP_KS_ARRAY[@]}"; do
        eval_top_ks_args+=" $k"
    done
    
    python run.py \
        --dataset "$dataset" \
        --corpus "$corpus" \
        --model "$MODEL" \
        --max-rounds "$MAX_ROUNDS" \
        --top-k "$TOP_K" \
        --eval-top-ks $eval_top_ks_args \
        --limit "$LIMIT" \
        --output "$output_filename"
    
    echo -e "${GREEN}Evaluation complete for $dataset_name_log using model $MODEL${NC}"
    echo ""
}

# Loop through each model specified
for MODEL in $MODELS_TO_RUN_STR; do
    # Validate RAG model
    if [[ "$MODEL" != "vanilla" && "$MODEL" != "agentic" && "$MODEL" != "light" ]]; then
        echo -e "${YELLOW}Invalid RAG model specified: $MODEL. Must be 'vanilla', 'agentic', or 'light'. Skipping.${NC}"
        continue
    fi

    echo -e "${GREEN}Processing evaluations for model: $MODEL${NC}"

    # HotpotQA dataset
    if should_run_dataset "hotpotqa"; then
        echo -e "${YELLOW}=== HotpotQA Dataset (Model: $MODEL) ===${NC}"
        run_evaluation \
            "dataset/hotpotqa.json" \
            "dataset/hotpotqa_corpus.json" \
            "${MODEL}_hotpotqa_evaluation.json"
    fi

    # MuSiQue dataset
    if should_run_dataset "musique"; then
        echo -e "${YELLOW}=== MuSiQue Dataset (Model: $MODEL) ===${NC}"
        run_evaluation \
            "dataset/musique.json" \
            "dataset/musique_corpus.json" \
            "${MODEL}_musique_evaluation.json"
    fi

    # 2WikiMultihopQA dataset
    if should_run_dataset "2wikimultihopqa"; then
        echo -e "${YELLOW}=== 2WikiMultihopQA Dataset (Model: $MODEL) ===${NC}"
        run_evaluation \
            "dataset/2wikimultihopqa.json" \
            "dataset/2wikimultihopqa_corpus.json" \
            "${MODEL}_2wikimultihopqa_evaluation.json"
    fi
done

echo -e "${GREEN}All specified evaluations complete!${NC}"
echo -e "Results saved in the $RESULTS_DIR directory."