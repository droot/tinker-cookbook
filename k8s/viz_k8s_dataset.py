import chz
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.format_colorized import format_colorized
try:
    from k8s.k8s_data import K8sDatasetBuilder
except ImportError:
    # Fallback for when running directly from k8s directory or without k8s package
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from k8s_data import K8sDatasetBuilder

@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3"
    dataset_path: str = "k8s-dataset.jsonl"
    max_length: int = 4096
    generate_sub_traj: bool = False
    batch_size: int = 4
    strip_thinking_from_history: bool = False

def run(cfg: Config):
    print(f"Loading dataset from {cfg.dataset_path}...")
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = renderers.get_renderer(cfg.renderer_name, tokenizer)
    # See https://tinker-docs.thinkingmachines.ai/rl/sequence-extension
    # for more details on why this is needed.
    renderer.strip_thinking_from_history = cfg.strip_thinking_from_history
    
    # Load dataset using our custom loader
    # Load dataset using our custom loader
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    builder = K8sDatasetBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=cfg.model_name,
            renderer_name=cfg.renderer_name,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
        ),
        data_files=cfg.dataset_path,
        generate_sub_traj=cfg.generate_sub_traj,
    )
    # We may need to manually inject this property since the builder normally gets it from config
    # actually ChatDatasetBuilder creates its own renderer from common_config
    if cfg.strip_thinking_from_history:
        # This is a bit tricky as the builder creates a fresh renderer
        # We might need to monkeypatch or just accept that we can't easily modify the renderer inside the builder
        # efficiently without changing the builder API or using a custom renderer name.
        # For now, let's assume the user uses a renderer name that handles this or we ignore it if mostly for viz
        print("Warning: strip_thinking_from_history is not directly supported via builder yet without custom renderer")

    ds_wrapper, _ = builder()
    dataset = ds_wrapper.hf_dataset
    print(f"Loaded {len(dataset)} examples.")
    
    # Iterate over batches
    for i in range(len(dataset)):
        try:
            batch_datums = ds_wrapper.get_batch(i)
        except IndexError:
            break
            
        if not batch_datums:
            break
            
        for datum in batch_datums:
            int_tokens = list(datum.model_input.to_ints())
            weights = datum.loss_fn_inputs["weights"].tolist()
            
            # Align weights with tokens if needed
            if len(weights) < len(int_tokens):
                weights = [0.0] * (len(int_tokens) - len(weights)) + weights
            elif len(weights) > len(int_tokens):
                weights = weights[:len(int_tokens)]
                
            print("-" * 80)
            print(format_colorized(int_tokens, weights, tokenizer))
            print("-" * 80)
            
            user_input = input("Press Enter for next example, 'q' to quit: ")
            if user_input.lower() == 'q':
                return

if __name__ == "__main__":
    chz.nested_entrypoint(run, allow_hyphens=True)
