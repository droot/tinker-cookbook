# k8s dataset


## Visualize dataset without sub trajectories.

```sh
python -m k8s.viz_k8s_dataset --model_name=Qwen/Qwen3-4B-Instruct-2507 --dataset_path=k8s/k8s-dataset.jsonl
```

## Visualize dataset with sub trajectories

```sh
python -m k8s.viz_k8s_dataset --model_name=Qwen/Qwen3-4B-Instruct-2507 --dataset_path=k8s/k8s-dataset.jsonl --strip_thinking_from_history=False --generate_sub_traj=True
```
