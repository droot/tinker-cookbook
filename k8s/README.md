# k8s dataset


## Visualize dataset without sub trajectories.

```sh
python -m k8s.viz_k8s_dataset --model_name=Qwen/Qwen3-4B-Instruct-2507 --dataset_path=k8s/k8s-dataset.jsonl
```

## Visualize dataset with sub trajectories

```sh
python -m k8s.viz_k8s_dataset --model_name=Qwen/Qwen3-4B-Instruct-2507 --dataset_path=k8s/k8s-dataset.jsonl --strip_thinking_from_history=False --generate_sub_traj=True
```

## Steps to train on k8s dataset

```sh
python -m tinker_cookbook.supervised.train --model_name="Qwen/Qwen3-4B-Instruct-2507" --log_path=/tmp/qwen3-4b-it-ft1 --dataset_builder=k8s.k8s_data.K8sDatasetBuilder --dataset_builder.common_config.model_name_for_tokenizer="Qwen/Qwen3-4B-Instruct-2507" --dataset_builder.common_config.renderer_name=qwen3 --dataset_builder.common_config.max_length=4096 --dataset_builder.common_config.batch_size=32 --dataset_builder.data_files=k8s/k8s-dataset.jsonl --dataset_builder.generate_sub_traj=True --num_epochs=1
```