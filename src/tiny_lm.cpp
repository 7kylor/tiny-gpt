// Tiny LM - One simple language model
// Word-level bigram: learns P(next_word | current_word)
// Produces readable text in seconds

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <iomanip>

static std::mt19937 g_rng(42);

class TinyLM {
public:
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;
    std::vector<std::vector<float>> logits;  // [vocab_size][vocab_size]
    size_t vocab_size = 0;
    
    // Tokenize text into words
    std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> words;
        std::string word;
        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t') {
                if (!word.empty()) {
                    words.push_back(word);
                    word.clear();
                }
                if (c == '\n') words.push_back("\n");
            } else {
                word += c;
            }
        }
        if (!word.empty()) words.push_back(word);
        return words;
    }
    
    void build_vocab(const std::string& text, size_t max_vocab = 5000) {
        auto words = tokenize(text);
        
        // Count word frequencies
        std::unordered_map<std::string, int> counts;
        for (const auto& w : words) counts[w]++;
        
        // Sort by frequency
        std::vector<std::pair<std::string, int>> sorted_words(counts.begin(), counts.end());
        std::sort(sorted_words.begin(), sorted_words.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Keep top words
        vocab_size = std::min(max_vocab, sorted_words.size());
        for (size_t i = 0; i < vocab_size; ++i) {
            word_to_id[sorted_words[i].first] = i;
            id_to_word.push_back(sorted_words[i].first);
        }
        
        // Initialize logits with small random values
        logits.resize(vocab_size, std::vector<float>(vocab_size, 0.0f));
        std::normal_distribution<float> dist(0.0f, 0.01f);
        for (auto& row : logits) {
            for (float& v : row) v = dist(g_rng);
        }
    }
    
    std::vector<int> encode(const std::vector<std::string>& words) {
        std::vector<int> ids;
        for (const auto& w : words) {
            auto it = word_to_id.find(w);
            if (it != word_to_id.end()) ids.push_back(it->second);
        }
        return ids;
    }
    
    std::string decode(const std::vector<int>& ids) {
        std::string text;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] >= 0 && ids[i] < static_cast<int>(vocab_size)) {
                const auto& word = id_to_word[ids[i]];
                if (word == "\n") {
                    text += "\n";
                } else {
                    if (i > 0 && id_to_word[ids[i-1]] != "\n") text += " ";
                    text += word;
                }
            }
        }
        return text;
    }
    
    // Get probabilities for next word given current
    std::vector<float> get_probs(int current) {
        std::vector<float> probs(vocab_size);
        float max_val = logits[current][0];
        for (size_t i = 1; i < vocab_size; ++i) {
            max_val = std::max(max_val, logits[current][i]);
        }
        float sum = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(logits[current][i] - max_val);
            sum += probs[i];
        }
        for (float& p : probs) p /= sum;
        return probs;
    }
    
    // Train on a batch of word pairs
    float train_step(const std::vector<int>& tokens, float lr) {
        float loss = 0.0f;
        
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            int curr = tokens[i];
            int next = tokens[i + 1];
            
            auto probs = get_probs(curr);
            loss -= std::log(probs[next] + 1e-10f);
            
            // Gradient descent: logits[curr] -= lr * (probs - one_hot(next))
            for (size_t j = 0; j < vocab_size; ++j) {
                float target = (j == static_cast<size_t>(next)) ? 1.0f : 0.0f;
                logits[curr][j] -= lr * (probs[j] - target);
            }
        }
        
        return loss / (tokens.size() - 1);
    }
    
    // Generate text
    std::string generate(const std::string& seed, size_t num_words, float temp = 1.0f) {
        std::vector<int> ids;
        
        // Find seed word
        auto it = word_to_id.find(seed);
        int curr = (it != word_to_id.end()) ? it->second : 0;
        ids.push_back(curr);
        
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < num_words; ++i) {
            // Apply temperature
            std::vector<float> temps(vocab_size);
            for (size_t j = 0; j < vocab_size; ++j) {
                temps[j] = logits[curr][j] / temp;
            }
            
            // Softmax
            float max_val = *std::max_element(temps.begin(), temps.end());
            float sum = 0.0f;
            for (float& v : temps) {
                v = std::exp(v - max_val);
                sum += v;
            }
            for (float& v : temps) v /= sum;
            
            // Sample
            float r = dist(g_rng);
            float cumsum = 0.0f;
            int next = 0;
            for (size_t j = 0; j < vocab_size; ++j) {
                cumsum += temps[j];
                if (cumsum >= r) {
                    next = j;
                    break;
                }
            }
            
            ids.push_back(next);
            curr = next;
        }
        
        return decode(ids);
    }
};

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open: " + path);
    std::stringstream buf;
    buf << file.rdbuf();
    return buf.str();
}

int main(int argc, char* argv[]) {
    std::cout << "=== Tiny LM ===\n";
    std::cout << "Word-level language model - produces readable text fast\n\n";
    
    std::string data_file = "data/tinyshakespeare.txt";
    size_t vocab_size = 3000;
    size_t max_steps = 5000;
    float lr = 1.0f;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) data_file = argv[++i];
        else if (arg == "--vocab" && i + 1 < argc) vocab_size = std::stoul(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) max_steps = std::stoul(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: tiny_lm [options]\n";
            std::cout << "  --data <file>   Training data\n";
            std::cout << "  --vocab <n>     Vocabulary size (default: 3000)\n";
            std::cout << "  --steps <n>     Training steps (default: 5000)\n";
            std::cout << "  --lr <f>        Learning rate (default: 1.0)\n";
            return 0;
        }
    }
    
    // Load data
    std::cout << "Loading: " << data_file << "\n";
    std::string text = read_file(data_file);
    std::cout << "Data: " << text.size() << " characters\n";
    
    // Build model
    TinyLM model;
    model.build_vocab(text, vocab_size);
    std::cout << "Vocab: " << model.vocab_size << " words\n";
    std::cout << "Params: " << model.vocab_size * model.vocab_size << "\n\n";
    
    // Tokenize
    auto words = model.tokenize(text);
    auto tokens = model.encode(words);
    std::cout << "Tokens: " << tokens.size() << "\n\n";
    
    // Train
    std::cout << "Training...\n";
    auto start = std::chrono::high_resolution_clock::now();
    std::uniform_int_distribution<size_t> dist(0, tokens.size() - 101);
    
    for (size_t step = 0; step < max_steps; ++step) {
        size_t idx = dist(g_rng);
        std::vector<int> batch(tokens.begin() + idx, tokens.begin() + idx + 100);
        
        float loss = model.train_step(batch, lr);
        
        if (step % 500 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            std::cout << "Step " << std::setw(4) << step 
                      << " | Loss: " << std::fixed << std::setprecision(3) << loss
                      << " | Time: " << std::setprecision(1) << elapsed << "s\n";
        }
        
        if (step % 1000 == 0 && step > 0) {
            std::cout << "\n--- Sample ---\n";
            std::cout << model.generate("the", 50, 1.0f) << "\n";
            std::cout << "--------------\n\n";
        }
    }
    
    // Final generation
    std::cout << "\n=== Generated Text ===\n\n";
    std::cout << model.generate("ROMEO:", 100, 0.9f) << "\n\n";
    std::cout << model.generate("What", 80, 0.9f) << "\n\n";
    std::cout << model.generate("I", 80, 0.9f) << "\n";
    
    return 0;
}
