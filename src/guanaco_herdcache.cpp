#include "guanaco/guanaco.h"
#include <algorithm>
#include <iostream>
#include <sys/mman.h>

namespace guanaco {

HerdCache::HerdCache(const HerdCacheConfig& config) : config_(config) {}

HerdCache::~HerdCache() {
    active_experts_.clear();
    lru_list_.clear();
}

std::shared_ptr<ExpertTensorBuffer> HerdCache::allocate_slot(int layer_idx, int expert_idx) {
    ExpertKey key{layer_idx, expert_idx};
    
    if (active_experts_.size() >= config_.max_active_experts) {
        evict_if_needed();
    }

    auto buf = std::make_shared<ExpertTensorBuffer>();
    buf->expert_id = expert_idx;
    buf->layer_idx = layer_idx;
    buf->is_ready = false;

    active_experts_[key] = buf;
    update_lru(key);
    
    std::cout << "[Guanaco Cache] Allocated slot for Layer " << layer_idx 
              << " Expert " << expert_idx << " (active: " << active_experts_.size() << ")" << std::endl;
    
    return buf;
}

std::shared_ptr<ExpertTensorBuffer> HerdCache::get_buffer(int layer_idx, int expert_idx) {
    ExpertKey key{layer_idx, expert_idx};
    auto it = active_experts_.find(key);
    if (it != active_experts_.end()) {
        update_lru(key);
        return it->second;
    }
    return nullptr;
}

bool HerdCache::is_pinned(int layer_idx, int expert_idx) const {
    ExpertKey key{layer_idx, expert_idx};
    return active_experts_.find(key) != active_experts_.end();
}

void HerdCache::pin_expert(int layer_idx, int expert_idx) {
    ExpertKey key{layer_idx, expert_idx};
    if (active_experts_.find(key) == active_experts_.end()) {
        allocate_slot(layer_idx, expert_idx);
    } else {
        update_lru(key);
    }
}

void HerdCache::unpin_expert(int layer_idx, int expert_idx) {
    ExpertKey key{layer_idx, expert_idx};
    auto it = active_experts_.find(key);
    if (it != active_experts_.end()) {
        if (config_.use_madvise && it->second && !it->second->data.empty()) {
            madvise(it->second->data.data(), it->second->data.size(), MADV_DONTNEED);
        }
        active_experts_.erase(it);
        
        auto lru_it = std::find(lru_list_.begin(), lru_list_.end(), key);
        if (lru_it != lru_list_.end()) {
            lru_list_.erase(lru_it);
        }
    }
}

void HerdCache::evict_if_needed() {
    while (active_experts_.size() >= config_.max_active_experts && !lru_list_.empty()) {
        evict_lru();
    }
}

void HerdCache::update_lru(const ExpertKey& key) {
    auto it = std::find(lru_list_.begin(), lru_list_.end(), key);
    if (it != lru_list_.end()) {
        lru_list_.erase(it);
    }
    lru_list_.push_back(key);
}

void HerdCache::evict_lru() {
    if (lru_list_.empty()) return;
    
    ExpertKey key = lru_list_.front();
    lru_list_.erase(lru_list_.begin());
    
    auto it = active_experts_.find(key);
    if (it != active_experts_.end()) {
        if (config_.use_madvise && it->second && !it->second->data.empty()) {
            madvise(it->second->data.data(), it->second->data.size(), MADV_DONTNEED);
        }
        std::cout << "[Guanaco Cache] Evicted Layer " << key.layer_idx 
                  << " Expert " << key.expert_idx << std::endl;
        active_experts_.erase(it);
    }
}

void HerdCache::apply_madvise(void* ptr, size_t size, bool will_need) {
    if (config_.use_madvise && ptr && size > 0) {
        madvise(ptr, size, will_need ? MADV_WILLNEED : MADV_DONTNEED);
    }
}

} // namespace guanaco