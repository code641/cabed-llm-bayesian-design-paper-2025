# CA-BED

**CA-BED: Conversation-Aware Bayesian Experimental Design**

Large Language Models (LLMs) excel at static reasoning tasks, yet their reliability diminishes in interactive scenarios where information must be actively acquired through questioning. A key challenge lies in asking questions that minimize uncertainty while simultaneously internalizing responses that may be ambiguous or partially specified. To address this, we propose Conversation-Aware Bayesian Experimental Design (CA-BED), an inference-time probabilistic dialogue planning framework that integrates Bayesian Experimental Design with LLM-based likelihood estimation to optimize question selection over multiple conversational turns. CA-BED models uncertainty continuously, anticipates potential answers, and propagates expected information gain through a simulated conversation tree, enabling more efficient and robust information acquisition. Across entity-deduction benchmarks, CA-BED yields an average 25\% improvement in success rates over direct prompting, with comparable gains relative to alternative information-seeking methods. It further achieves a 10.25\% average reduction in conversation length, indicating greater efficiency in information gathering. These results highlight CA-BED’s effectiveness as a principled framework for reliable interactive reasoning in real-world settings.

## Setup

1. Install [uv](https://docs.astral.sh/uv/)
2. Clone the the `ca-bed` package
3. Run `uv sync` in the root directory
4. Create a `.env` file and populate `DEEPSEEK_KEY`

## Use

Run experiments with `uv run main.py <task_name> [options]`, which implements CA-BED, CA-BED + Answer-Planning, UoT, and Direct Prompting.

### Available Tasks

[**Detective Cases**](https://github.com/tmlr-group/AR-Bench)

| Task Name                  | Description                                    |
| -------------------------- | ---------------------------------------------- |
| `detective_direct`         | Direct prompting baseline (no reasoning tree). |
| `detective_uot`            | Uncertainty of Thoughts (UoT)                  |
| `detective_bayesian`       | CA-BED                                         |
| `detective_bayesian_multi` | CA-BED + Answer-Planning                       |

[**Twenty Questions**](https://github.com/zhiyuanhubj/UoT)

| Task Name                | Description                                    |
| ------------------------ | ---------------------------------------------- |
| `twentyq_direct`         | Direct prompting baseline (no reasoning tree). |
| `twentyq_uot`            | Uncertainty of Thoughts (UoT)                  |
| `twentyq_bayesian`       | CA-BED                                         |
| `twentyq_bayesian_multi` | CA-BED + Answer-Planning                       |

### Common Arguments

| Argument                 | Type    | Default               | Description                                  |
| ------------------------ | ------- | --------------------- | -------------------------------------------- |
| `--questioner_model`     | `str`   | `"deepseek-chat"`     | Model key for the questioner.                |
| `--answerer_model`       | `str`   | `"deepseek-reasoner"` | Model key for the answerer.                  |
| `--start_idx`            | `int`   | `0`                   | Start index for dataset subset.              |
| `--end_idx`              | `int`   | `10`                  | End index for dataset subset.                |
| `--conversation_depth`   | `int`   | `20`                  | Maximum conversation depth.                  |
| `--max_concurrent`       | `int`   | `6`                   | Maximum concurrent tasks.                    |
| `--clustering_threshold` | `float` | `1.0`                 | Threshold for question clustering.           |
| `--shared_cluster`       | `flag`  | `off`                 | Use a shared question cluster for all runs.  |
| `--output_dir`           | `str`   | `logs/<timestamp>`    | Directory where results are saved.           |
| `--sharpness_constant`   | `float` | `0.4`                 | λ constant to penalize biased questions.     |
| `--min_probability`      | `float` | `1/25000`             | Minimum probability cutoff to prune answers. |

### Additional Arguments for Tree-Based Methods

| Argument                 | Type    | Default | Description                                             |
| ------------------------ | ------- | ------- | ------------------------------------------------------- |
| `--max_question_nodes`   | `int`   | `2`     | Maximum number of question nodes per turn.              |
| `--max_lookahead_depth`  | `int`   | `3`     | Lookahead search depth for planning.                    |
| `--confidence_threshold` | `float` | `0.8`   | Confidence threshold for terminating                    |
| `--estimator_confidence` | `float` | `0.7`   | Ɛ confidence constant for the LLM likelihood estimator. |

### Output

Each experiment by default creates a timestamped directory under `logs/`, containing:

- `logs.log` - full runtime logs
- `<idx>_run.json` - serialized run record
- `<idx>_cluster.json`/`<idx>_cluster.voy` - saved likelihood clustering/caching state

## Evaluation and Analysis

After running one or more experiments, you can evaluate and compare their results using `uv run eval.py --paths <path_to_experiment_dir> [<path_to_another_experiment_dir> ...]`. This reads the `*_run.json` files generated by each experiment, computes performance statistics, and summarizes them in a comparison table.

### Summary Statistics Explained

Each experiment directory is evaluated independently. For each run, the following metrics are computed:

Here is the markdown for that table:

| Metric              | Description                                                       |
| ------------------- | ----------------------------------------------------------------- |
| Top-1               | Whether the model's most likely guess matches the correct answer. |
| Top-3               | Whether the correct answer appears in the top-3 guesses.          |
| Conversation Length | Number of question-answer turns in the dialogue.                  |
| Start / End Time    | Used to measure total runtime duration.                           |
| Token Usage         | Input/output token counts for both questioner and answerer.       |

### Arguments

| Argument                    | Type    | Default    | Description                                       |
| --------------------------- | ------- | ---------- | ------------------------------------------------- |
| `-p`, `--paths`             | `Path`  | _required_ | One or more experiment directories to evaluate.   |
| `--questioner-input-price`  | `float` | `0.28`     | Price per 1M tokens for questioner input tokens.  |
| `--questioner-output-price` | `float` | `0.42`     | Price per 1M tokens for questioner output tokens. |
| `--answerer-input-price`    | `float` | `0.28`     | Price per 1M tokens for answerer input tokens.    |
| `--answerer-output-price`   | `float` | `0.42`     | Price per 1M tokens for answerer output tokens.   |
