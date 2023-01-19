#include"net/threads.h"

size_t SafeQueue::size(){
    std::unique_lock<std::mutex> lock(mutex_);
    return items_.size();
}

void SafeQueue::wait_ready(std::unique_lock<std::mutex> &lock, int waitMs){
    if (exit_ || !items_.empty()) {
        return;
    }
    if (waitMs == wait_infinite) {
        condition_.wait(lock, [this] { return exit_ || !items_.empty(); });
    } else if (waitMs > 0) {
        auto tp = std::chrono::steady_clock::now() + std::chrono::milliseconds(waitMs);
        while (condition_.wait_until(lock, tp) != std::cv_status::timeout && items_.empty() && !exit_) {
        }
    }
}

bool SafeQueue::push(Task && t){
    std::unique_lock<std::mutex> lock(mutex_);
    if(exit_ || (capacity_ && items_.size()>=capacity_))
        return false;
    items_.push_back(std::move(t));
    condition_.notify_one();
    return true;
}

bool SafeQueue::pop(Task * t, int waitMs){
    std::unique_lock<std::mutex> lock(mutex_);
    wait_ready(lock, waitMs);
    if(items_.empty())
        return false;
    *t = std::move(items_.front());
    items_.pop_front();
    return true;
}

void SafeQueue::exit(){
    {
        std::unique_lock<std::mutex> lock(mutex_);
        exit_=true;
    }
    condition_.notify_all();
}

bool SafeQueue::empty(){
    std::unique_lock<std::mutex> lock(mutex_);
    return items_.empty();
}

Thread_Pool::Thread_Pool(size_t threads, size_t taskCapacity):tasks_(taskCapacity){
    for(size_t i=0;i<threads;++i){
        workers_.emplace_back([this]{
            for(;;){
                if(tasks_.exited() && tasks_.empty())
                    return;
                Task task;
                if (tasks_.pop(&task)) {
                    task();
                }
            }
        });
    }
}

Thread_Pool::~Thread_Pool(){
    tasks_.exit();
    for(std::thread& worker: workers_)
        worker.join();
}

bool Thread_Pool::addTask(Task &&task){
    return tasks_.push(std::move(task));
}