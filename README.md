# Preliminary Survey on Foundation Language Models

### [Paper](https://github.com/uf-eel6825-sp23/final-project-code-01-vyom/blob/main/doc/Preliminary_Survey_on_Foundation_Language_Models.pdf) | [Slides](https://docs.google.com/presentation/d/1cuZVWwlQMdkUFz0OY3Rl6bq2NeZnun0WJsWCYAMGx6M/edit?usp=sharing) | [Video](https://youtu.be/A7ZRedh2nuE)

## Abstract
Language models are essential components of natural language processing that can capture general representations of language from large-scale text corpora. In recent years, various language models have been proposed, ranging from shallow word embeddings to deep contextual encoders, with different pre-training tasks, training frameworks, adaptation methods, and evaluation benchmarks. The year 2022 saw a surge in the development of large generative models, referred to as "foundation models," that can perform multiple tasks by training on a general unlabeled dataset. However, there is a need for a comprehensive survey that can link these models and contrast their advantages and disadvantages. In this paper, we present a systematic survey of foundation language models that aims to connect various models based on multiple criteria, such as representation learning, model size, task capabilities, research questions, and practical task capabilities. We also conduct experiments on a subset of SuperGLUE tasks using six representative models from different architecture families and contrast their performance and efficiency. Moreover, we suggest some future directions that can be pursued to improve the robustness and comprehensiveness of this survey.
## About The Project
The survey constitues of what aspects of languagels can be considered to differentiate them. We then go in detail for each aspect one by one, and discuss their respective sub-categories. We then, propose some prilimanry model catelogue talking about how they are dfferent from each other as well connected to each other based on pre-training task, architecture, model size, and other factors. Then, we perform a prilimanry evaluation on some of the few select models from the catelogue. We then, discuss the results and the future work. More details can be found in the [paper](https://github.com/uf-eel6825-sp23/final-project-code-01-vyom/blob/main/doc/Preliminary_Survey_on_Foundation_Language_Models.pdf). The dataset detail, and the results are provided in the following sections. After that, we provide the details about the system requirements, and the steps to reproduce the results.
### Dataset Details

The following table shows the details about each of the dataset used to train, and evaluate the models. Each dataset is a part of the SuperGLUE benchmark. The SuperGLUE benchmark is a collection of 9 different tasks that are used to evaluate the performance of language models. More details can be found in the SuperGLUE [paper](https://arxiv.org/abs/1905.00537).


| Name                                                                                 | Base Dataset | Task                                     | Metric      |
| ------------------------------------------------------------------------------------ | ------------ | ---------------------------------------- | ----------- |
| [Winograd Schema Challenge](https://arxiv.org/abs/1905.00537)                        | Super GLUE   | Coreference Resolution                   | Accuracy    |
| [Boolean Questions](https://arxiv.org/abs/1905.00537)                                | Super GLUE   | Boolean Question Answering               | Accuracy    |
| [Commitment Bank](https://arxiv.org/abs/1905.00537)                                  | Super GLUE   | Natural Language Inferencing             | Accuracy/F1 |
| [PASCAL: Recognizing Textual Entailment Challenge](https://arxiv.org/abs/1905.00537) | Super GLUE   | Binary Textual Entailment                | Accuracy    |
| [Words in Context](https://arxiv.org/abs/1905.00537)                                 | Super GLUE   | Word Sense Disambiguation                | Accuracy    |
| [Choice of Plausible Alternatives](https://arxiv.org/abs/1905.00537)                 | Super GLUE   | Open-domain Commonsense Causal Reasoning | Accuracy    |

### Results

Here we present the results of the models trained on select tasks from the SuperGLUE benchmark. We show a select models which are categorized based on their architecture i.e. encoder-only, encoder-decoder, and decoder-only. The results are shown in the table below.

| Dataset/Model | [RoBERTa](https://arxiv.org/abs/1907.11692) | [DeBERTa](https://arxiv.org/abs/2006.03654) | [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [Transformer-XL](https://arxiv.org/abs/1901.02860) | [Bart](https://arxiv.org/abs/1910.13461) | [T5](https://arxiv.org/abs/1910.10683) |
| ------------- | ------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------- | -------------------------------------- |
| BooQ (acc.)   | 0.6217                                      | **`0.8685`**                                | 0.7695                                                                                                                      | 0.6714                                             | 0.85107                                  | 0.7877                                 |
| RTE (acc.)    | 0.527                                       | 0.8591                                      | 0.7184                                                                                                                      | 0.570                                              | 0.8519                                   | **`0.8592`**                           |
| COPA (acc.)   | 0.55                                        | 0.58                                        | 0.6                                                                                                                         | 0.57                                               | 0.56                                     | **`0.66`**                             |
| WIC (acc.)    | 0.5                                         | **`0.7116`**                                | 0.6912                                                                                                                      | 0.5360                                             | 0.6834                                   | 0.6914                                 |
| WSC (acc.)    | 0.6302                                      | **`0.6347`**                                | 0.6442                                                                                                                      | 0.6340                                             | 0.6346                                   | 0.5194                                 |
| CB (acc./F1)  | 0.8081 / 0.765                              | 0.6785 / 0.4740                             | 0.7857 / 0.6398                                                                                                             | 0.7321 / 0.5113                                    | 0.6785 / 0.4700                          | **`0.875 / 0.7854`**                   |

The results indicate that the best modesl are DeBERTa and T5. The best model for each task is highlighted in bold. These models are provided on canvas.

## Getting Started
### System & Requirements

All the experiments were run on a single `NVIDIA A100 GPU with 80GB memory on HiperGator3`. The code is written in Python 3. The installation guide goes over how to setup the environment with the required dependencies. The dataset are downloaded from the huggingface datasets library. The models are trained using the huggingface transformers library specifically, the `pytorch` framework.

### Extracting Best Model Weights

1.  Change the directory to `blue/eel6825/<username>/`
  ```sh
  cd /blue/eel6825/<username>/
  ```

2. Upload the model weights downloaded from [google drive](https://drive.google.com/file/d/1ZxTwfWYqdUWvRURHl6T3gZ3Y9Ygb3lPv/view?usp=share_link) (8 GB) and save them in the `blue/eel6825/<username>/` directory. The model weights are saved in the `.tar.gz.zip` format. You can extract them using the following command:
  ```sh
  unzip models.tar.gz.zip
  tar -xvzf models.tar.gz
  ```
  This will extract all the best models in the `blue/eel6825/<username>/models` directory. They are saved as `$MODEL_NAME/$TASK_NAME` format. For example, the best model for `deberta` on `boolq` task is saved in the `blue/eel6825/<username>/models/deberta/boolq` directory.
  The best models are those models which are highlighted in bold in the results table.

### Installation

1. Clone the repo
  ```sh
  git clone https://github.com/catiaspsilva/README-template.git
  ```

2. Change the directory to the `src` folder
  ```sh
  cd src
  ```

3. Setup (and activate) your environment on a GPU enabled node
  ```sh
  srun --ntasks=1 --cpus-per-task=2 --mem=15gb --account eel6825 --qos eel6825 --partition=gpu --gres=gpu:a100:1 --time=10:00:00 --pty bash -i
  ml conda
  mamba env create -f requirements.yml
  ```

4. Activate the environment you just created on a GPU enabled node
  ```sh
  mamba activate llm
  ```
## Usage
### Training

After installation, the model can be trained using a batch job on HiperGator3. The following command will submit a batch job to the cluster with the necessary parameters to train the model. The batch job will run for 24 hours and will send an email notification when the job is completed.
```sh
sbatch <model_name>.sh
```

Note:
- Here, the `<model_name>` is the name of the model you want to train. For example, to train the `roberta` model you would run `sbatch roberta.sh`. All the model scripts with their parameters are listed in the following table. Important thing to note here is that we dont finetune `t5` as it is already `pre-finetuned` on Huggingface.
- For each script, change the `--output_dir` to the directory where you want to save the model checkpoints. The output directory should be in the `blue` directory as it has more space than the `home` directory. The creation of the output directory is handled by the script provided we have given a path. For example, if you want to save the model checkpoints in the `blue/eel6825/<username>/output/$MODEL_NAME/$TASK_NAME` directory, you would change the `--output_dir` to `--output_dir=/blue/eel6825/<username>/output/$MODEL_NAME/$TASK_NAME`. It is recommended to use this format so one can perform quick training as well as evaluation on the same model.
- For each script, change the `#SBATCH --mail-user` to specific the mail id where you want to receive the email notifications for the training job. For example, if you want to receive the email notifications to your `ufl.edu` email id, you would change the `--mail-user` to `--mail-user=<username>@ufl.edu`.
- In `train.py` script, change the following parameters:
  ```
  os.environ["TRANSFORMERS_CACHE"] = "/blue/eel6825/<username>/.cache"
  os.environ["HF_HOME"] = "/blue/eel6825/<username>/.cache"
  os.environ["XDG_CACHE_HOME"] = "/blue/eel6825/<username>/.cache"
  ```
  These parameters are used to save the model checkpoints in the cache directory. Here, `<username>` is your username on HiperGator3. This is important, we want to store any new donwdloaded pre-trained model to `blue` directory as it has more space than the `home` directory.

Alternatively, you can get a interactive session on the cluster and run the training script.
```sh
srun --ntasks=1 --cpus-per-task=2 --mem=15gb --account eel6825 --qos eel6825 --partition=gpu --gres=gpu:a100:1 --time=10:00:00 --pty bash -i
ml conda
mamba activate llm
export MODEL_NAME=<model_name> # e.g. roberta
export TASK_NAME=<task_name> # e.g. wic
python train.py \
  --model_name_or_path transfo-xl-wt103 \
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_train_batch_size 16 \
  --learning_rate 2.5e-4 \
  --num_train_epochs 10 \
  --seed 42 \
  --weight_decay 0.01 \
  --classifier_dropout 0.1 \
  --clip 0.25 \
  --checkpointing_steps epoch \
  --output_dir /blue/eel6825/<username>/output/$MODEL_NAME/$TASK_NAME/ # change the output directory
```
The above script runs transformer-xl on any of the datasets. The `--model_name_or_path` parameter can be changed to any of the models from the respective `<model-name>.sh` script. For example, if you want to run `roberta` on `wic` task, you would change the `--model_name_or_path` to `roberta-large`, `$TASK_NAME` to `wic`, `$MODEL_NAME` to `roberta`, as described in `roberta.sh` script. This is true except for the `t5` model as we use the `t5-large` model for all the tasks.

| Model       | # of Parameters (Million) |
| ----------- | ------------------------- |
| `deberta`   | 350                       |
| `roberta`   | 355                       |
| `gpt2`      | 355                       |
| `transfoxl` | 355                       |
| `bart`      | 400                       |
| t5          | 770                       |

### Evaluation

After training, the models can be evaluated in 3 ways.

Firstly, we can evaluate all the models using the following command. This will evaluate all the models on all the tasks and save the results in the `results` directory.
```sh
sbatch eval_all.sh
```

Secondly, if we want to evaluate the best models for each task, we can use the following command. This will evaluate the best models for each task and save the results in the `results` directory.
```sh
sbatch eval_best.sh
```

Note:
- For both the eval script, change the `#SBATCH --mail-user` to specific the mail id where you want to receive the email notifications for the training job. For example, if you want to receive the email notifications to your `ufl.edu` email id, you would change the `--mail-user` to `--mail-user=<username>@ufl.edu`.
- For both the eval script, change the `--model_name_or_path` to the path where the model checkpoints are saved. For example, if you want to evaluate the `roberta` model on the `wic` task, you would change the `--model_name_or_path` to `--model_name_or_path=/blue/eel6825/<username>/output/$MODEL_NAME/$TASK_NAME/`. Here, `<username>` is your username on HiperGator3. If you are using the downloaded models, give the path to the downloaded model.
- For both the eval script, change the `--output_dir` to the directory where you want to save the results. The output directory should be in the `blue` directory as it has more space than the `home` directory. The creation of the output directory is handled by the script provided we have given a path. For example, if you want to save the results in the `blue/eel6825/<username>/results/$MODEL_NAME/$TASK_NAME` directory, you would change the `--output_dir` to `--output_dir=/blue/eel6825/<username>/results/$MODEL_NAME/$TASK_NAME`. 
- For the `eval_best.sh` do not change the MODEL_NAME, as it takes the best model for each task. In this case, it is important the the trained model were saved as follows: `blue/eel6825/<username>/output/$MODEL_NAME/$TASK_NAME/`. Here, `<username>` is your username on HiperGator3.
- In `eval.py` script, change the following parameters:
  ```
  os.environ["TRANSFORMERS_CACHE"] = "/blue/eel6825/<username>/.cache"
  os.environ["HF_HOME"] = "/blue/eel6825/<username>/.cache"
  os.environ["XDG_CACHE_HOME"] = "/blue/eel6825/<username>/.cache"
  ```
  These parameters are used to save the model checkpoints incase you download an huggingface model instead of using the provided model in the cache directory. Here, `<username>` is your username on HiperGator3. This is important, we want to store any new donwdloaded pre-trained model to `blue` directory as it has more space than the `home` directory.


Final alternative if we want to evaluate a specific model on a specific task, we can use the following command. This will evaluate the model on the task and save the results in the `results` directory.
```sh
srun --ntasks=1 --cpus-per-task=2 --mem=15gb --account eel6825 --qos eel6825 --partition=gpu --gres=gpu:a100:1 --time=10:00:00 --pty bash -i
ml conda
mamba activate llm
export MODEL_NAME=<model_name> # e.g. roberta
export TASK_NAME=<task_name> # e.g. wic
python eval.py \
  --model_name_or_path /blue/eel6825/<username>/output/$MODEL_NAME/$TASK_NAME/ \ # change the model path
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_eval_batch_size 16 \
  --output_dir /blue/eel6825/<username>/results/$MODEL_NAME/$TASK_NAME/ # change the output directory
```

Look at the `results` directory to see the results for every model on every task. More details on how the project structure is organized can be found in the [video](https://youtu.be/A7ZRedh2nuE).

## Roadmap

Detailed roadmap on improvements for this can be found in the future work section of the [paper](https://github.com/uf-eel6825-sp23/final-project-code-01-vyom/blob/main/doc/Preliminary_Survey_on_Foundation_Language_Models.pdf).
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Authors

[Vyom Pathak](https://01-vyom.github.io/) - [@stancosmos01](https://twitter.com/stancosmos01) - angerstick3@gmail.com

Project Link: [https://github.com/uf-eel6825-sp23/final-project-code-01-vyom](https://github.com/uf-eel6825-sp23/final-project-code-01-vyom)


## Acknowledgements

Some part of the model training, and evaluation has been borrowed from the [transformers finetuning tutorials](https://huggingface.co/docs/transformers/training). We thank the University of Florida for providing us with the access to A100 80GB GPUs on [HiperGator3.0](https://www.rc.ufl.edu/about/hipergator/) for training, as well as evaluation of models.

## License

Licensed under the [MIT License](LICENSE.md).
