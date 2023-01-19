#ifndef THREADS_H
#define THREADS_H

#include<thread>
#include<vector>
#include<mutex>
#include<atomic>
#include<list>
#include<condition_variable>
#include"net/utils.h"
#include"net/def.h"


class SafeQueue: private noncopyable
{
private:
    static const int wait_infinite = std::numeric_limits<int>::max();
    std::mutex mutex_;
    std::atomic<bool> exit_;
    std::list<Task> items_;
    std::condition_variable condition_;
    size_t capacity_;
public:
    SafeQueue(size_t capacity=0, bool exit=false): capacity_(capacity),exit_(exit){}

    size_t size();

    void wait_ready(std::unique_lock<std::mutex> &lk, int waitMs);
    bool push(Task && t);
    bool pop(Task * t, int waitMs=wait_infinite);
    void exit();
    bool exited() { return exit_; }
    bool empty();
};



class Thread_Pool{

private:
    std::vector<std::thread> workers_;
    SafeQueue tasks_;

public:
    Thread_Pool(size_t threads, size_t taskCapacity = 0);
    ~Thread_Pool();

    bool addTask(Task &&task);
    bool addTask(Task &task) { return addTask(Task(task)); }

};





#endif