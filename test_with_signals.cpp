#include <microgpt/microgpt.h>
#include <iostream>
#include <csignal>
#include <cstring>
#include <execinfo.h>
#include <unistd.h>

void signal_handler(int sig) {
    std::cerr << "\n=== CAUGHT SIGNAL " << sig << " (" << strsignal(sig) << ") ===" << std::endl;
    
    // Print backtrace
    void* array[50];
    size_t size = backtrace(array, 50);
    
    std::cerr << "Backtrace (" << size << " frames):" << std::endl;
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    
    std::exit(1);
}

using namespace microgpt;

int main() {
    // Install signal handlers
    signal(SIGSEGV, signal_handler);
    signal(SIGABRT, signal_handler);
    
    std::cout << "Starting test with signal handlers..." << std::endl;
    
    try {
        ValueStorage storage;
        
        // Create 2 logits
        std::cout << "Creating logits..." << std::endl;
        Value* l1 = storage.store(Value(1.0));
        Value* l2 = storage.store(Value(2.0));
        
        std::vector<Value*> logits = {l1, l2};
        
        std::cout << "Calling softmax..." << std::endl;
        auto probs = softmax(logits, storage);
        
        std::cout << "Softmax done. probs[0] = " << probs[0]->data << std::endl;
        std::cout << "Softmax done. probs[1] = " << probs[1]->data << std::endl;
        
        std::cout << "Calling backward on probs[1]..." << std::endl;
        probs[1]->backward();
        
        std::cout << "SUCCESS! Backward completed." << std::endl;
        std::cout << "l1->grad = " << l1->grad << std::endl;
        std::cout << "l2->grad = " << l2->grad << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
