#include <guanaco/guanaco.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

int main(int argc, char** argv) {
    std::string model_path = argc > 1 ? argv[1] : "model.gguf";
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Guanaco - MoE Disk-Streaming Engine  " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << std::endl;

    guanaco::GuanacoExecutorConfig config;
    config.model_path = model_path;
    config.max_active_experts = 2;
    config.use_io_uring = true;
    config.use_madvise = true;

    guanaco::GuanacoExecutor engine(config);
    
    if (!engine.initialize()) {
        std::cerr << "[ERROR] Failed to initialize Guanaco engine" << std::endl;
        return 1;
    }

    const auto& model_config = engine.loader()->get_model_config();
    std::cout << "[Guanaco] Model: " << model_config.architecture << std::endl;
    std::cout << "[Guanaco] Total experts: " << model_config.num_experts 
              << ", top_k: " << model_config.num_experts_per_tok
              << ", layers: " << model_config.num_hidden_layers
              << ", hidden: " << model_config.hidden_size << std::endl;
    std::cout << "[Guanaco] Manifest entries: " << engine.loader()->get_manifest().size() << std::endl;
    
    for (const auto& entry : engine.loader()->get_manifest()) {
        std::cout << "  Layer " << entry.layer_idx 
                  << " (" << entry.tensor_name 
                  << "): " << entry.byte_size / 1024 / 1024 << " MB";
        if (entry.num_experts_in_tensor > 1) {
            double per_expert_mb = static_cast<double>(entry.expert_slice_size) / 1024.0 / 1024.0;
            std::cout << " (" << entry.num_experts_in_tensor 
                      << " experts, " << per_expert_mb << " MB/expert)";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "[Guanaco] Starting token processing simulation..." << std::endl;
    
    std::vector<float> hidden_states(model_config.hidden_size > 0 ? model_config.hidden_size : 4096, 0.1f);
    
    engine.set_router_callback([](int layer_idx, const std::vector<int>& experts) {
        std::cout << "[Guanaco Router] Layer " << layer_idx 
                  << " selected " << experts.size() << " experts: ";
        for (int e : experts) std::cout << e << " ";
        std::cout << std::endl;
    });

    for (int token_idx = 0; token_idx < 5; ++token_idx) {
        std::cout << "\n--- Token " << token_idx << " ---" << std::endl;
        engine.evaluate_token(hidden_states);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\n[Guanaco] Simulation complete" << std::endl;
    std::cout << "[Guanaco] Cache stats: " << engine.cache()->active_count() << " active experts" << std::endl;
    
    return 0;
}