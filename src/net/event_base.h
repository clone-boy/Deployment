#ifndef EVENT_BASE_H
#define EVENT_BASE_H

#include"net/utils.h"
#include"net/channel.h"
#include"net/threads.h"
#include"net/def.h"
#include<unistd.h>
#include"net/logging.h"
#include<string.h>
#include<map>
#include"net/conn.h"

class TimerRepeatable {
public:
    int64_t at;  // current timer timeout timestamp
    int64_t interval;  //重复间隔
    TimerId timerid;
    Task cb;
};

class EventBase{
public:
    std::unique_ptr<EventsImp> imp_;
    //指定任务队列的大小，0无限制
    EventBase(int taskCapacity=0);
    ~EventBase();
    //处理已到期的事件,waitMs表示若无当前需要处理的任务，需要等待的时间
    void loop_once(int waitMs);
    //进入事件处理循环
    void loop();
    //取消定时任务，若timer已经过期，则忽略
    bool cancel(TimerId timerid);

    //添加定时任务，interval=0表示一次性任务，否则为重复任务，时间为毫秒
    TimerId runAt(int64_t milli, const Task &task, int64_t interval = 0) { return runAt(milli, Task(task), interval); }
    TimerId runAt(int64_t milli, Task &&task, int64_t interval = 0);

    //添加延时任务，milli之后进行
    TimerId runAfter(int64_t milli, const Task &task, int64_t interval = 0) { return runAt(utils::steadyMilli() + milli, Task(task), interval); }
    TimerId runAfter(int64_t milli, Task &&task, int64_t interval = 0) { return runAt(utils::steadyMilli() + milli, std::move(task), interval); }

    //退出事件循环
    EventBase &exit();
    //是否已退出
    bool exited();
    //唤醒事件处理
    void wakeup();
    //添加任务
    void safeCall(Task &&task);
    void safeCall(const Task &task) { safeCall(Task(task)); }


    //分配一个事件派发器
    EventBase *allocBase() {return this;}
};


class EventsImp{
private:
    std::atomic<bool> exit_;
    SafeQueue tasks_;
    int wakeupFds_[2]; //利用管道进行唤醒

    int nextTimeout_; //下次超时时间

    std::map<TimerId, TimerRepeatable> timerReps_;  //重复任务
    std::map<TimerId, Task> timers_;  //定时器任务
    std::atomic<int64_t> timerSeq_;  //任务序列值

public:
    EventBase* base_;
    Epoller* epoller_;
    std::set<TcpConnPtr> reconnectConns_;
    
    EventsImp(EventBase* base, int taskCapacity);
    ~EventsImp();

    void init();

    EventBase &exit();
    bool exited() { return exit_; }
    void safeCall(Task &&task);
    void loop();
    void loop_once(int waitMs);
    void wakeup(); //当唤醒时就会执行任务队列里的所有任务

    void handleTimeouts();
    bool cancel(TimerId timerid);
    TimerId runAt(int64_t milli, Task &&task, int64_t interval);

    void refreshNearest(const TimerId *tid = NULL);
    void repeatableTimeout(TimerRepeatable *tr);
};









#endif