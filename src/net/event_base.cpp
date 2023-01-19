#include"net/event_base.h"
#include"net/net.h"
#include<fcntl.h>

EventBase::EventBase(int taskCapacity){
    imp_.reset(new EventsImp(this,taskCapacity));
    imp_->init();
}

EventBase::~EventBase(){
}

void EventBase::loop_once(int waitMs){
    imp_->loop_once(waitMs);
}

void EventBase::loop(){
    imp_->loop();
}

bool EventBase::cancel(TimerId timerid){
    return imp_ && imp_->cancel(timerid);
}

TimerId EventBase::runAt(int64_t milli, Task &&task, int64_t interval){
    return imp_->runAt(milli, std::move(task),interval);
}

EventBase & EventBase::exit(){
    return imp_->exit();
}

bool EventBase::exited(){
    return imp_->exited();
}

void EventBase::wakeup(){
    imp_->wakeup();
}

void EventBase::safeCall(Task &&task){
    imp_->safeCall(std::move(task));
}



EventsImp::EventsImp(EventBase* base, int taskCapacity): base_(base), epoller_(createEpoller()), exit_(false), tasks_(taskCapacity), timerSeq_(0), nextTimeout_(1 << 30) {}

EventsImp::~EventsImp(){
    delete epoller_;
    ::close(wakeupFds_[1]);
}

void EventsImp::init(){
    int r = pipe(wakeupFds_);
    fatalif(r, "pipe failed %d %s", errno, strerror(errno));
    r = net::addFdFlag(wakeupFds_[0], FD_CLOEXEC);
    fatalif(r, "addFdFlag failed %d %s", errno, strerror(errno));
    r = net::addFdFlag(wakeupFds_[1], FD_CLOEXEC);
    fatalif(r, "addFdFlag failed %d %s", errno, strerror(errno));
    trace("wakeup pipe created %d %d", wakeupFds_[0], wakeupFds_[1]);
    Channel *ch = new Channel(base_, wakeupFds_[0], kReadEvent);
    ch->onRead([=] {
        char buf[1024];
        int r = ch->fd() >= 0 ? ::read(ch->fd(), buf, sizeof buf) : 0;
        if (r > 0) {
            Task task;
            while (tasks_.pop(&task, 0)) {
                task();
            }
        } else if (r == 0) {//imp析构后写管道关闭
            delete ch; //读通道通过channel关闭
        } else if (errno == EINTR) {
        } else {
            fatal("wakeup channel read error %d %d %s", r, errno, strerror(errno));
        }
    });
}

EventBase & EventsImp::exit() {
    exit_ = true;
    wakeup();
    return *base_;
}

void EventsImp::safeCall(Task &&task) {
    tasks_.push(std::move(task));
    wakeup();
}

void EventsImp::loop(){
    while (!exit_)
        loop_once(10000);
    timerReps_.clear();
    timers_.clear();
    for (auto recon : reconnectConns_) {  //重连的连接无法通过channel清理，因此单独清理
        recon->cleanup(recon);
    }
    loop_once(0);
}

void EventsImp::loop_once(int waitMs) {
    epoller_->loop_once(std::min(waitMs, nextTimeout_));
    handleTimeouts(); //处理超时
}

void EventsImp::wakeup() {
    int r = write(wakeupFds_[1], "", 1);
    fatalif(r <= 0, "write error wd %d %d %s", r, errno, strerror(errno));
}

void EventsImp::handleTimeouts(){
    int64_t now = utils::steadyMilli();
    TimerId tid{now, 1L << 62};
    while (timers_.size() && timers_.begin()->first < tid) {
        Task task = std::move(timers_.begin()->second);
        timers_.erase(timers_.begin());
        task();
    }
    refreshNearest();
}

bool EventsImp::cancel(TimerId timerid){
    if (timerid.first < 0) {
        auto p = timerReps_.find(timerid);
        auto ptimer = timers_.find(p->second.timerid);
        if (ptimer != timers_.end()) {
            timers_.erase(ptimer);
        }
        timerReps_.erase(p);
        return true;
    } else {
        auto p = timers_.find(timerid);
        if (p != timers_.end()) {
            timers_.erase(p);
            return true;
        }
        return false;
    }
}

TimerId EventsImp::runAt(int64_t milli, Task &&task, int64_t interval){
    if (exit_) {
        return TimerId();
    }
    if (interval) {
        TimerId tid{-milli, ++timerSeq_};
        TimerRepeatable &rtr = timerReps_[tid];
        rtr = {milli, interval, {milli, ++timerSeq_}, std::move(task)};
        TimerRepeatable *tr = &rtr;
        timers_[tr->timerid] = [this, tr] { repeatableTimeout(tr); };
        refreshNearest(&tr->timerid);
        return tid;
    } else {
        TimerId tid{milli, ++timerSeq_};
        timers_.insert({tid, move(task)});
        refreshNearest(&tid);
        return tid;
    }
}

void EventsImp::refreshNearest(const TimerId *tid) {
    if (timers_.empty()) {
        nextTimeout_ = 1 << 30;
    } else {
        const TimerId &t = timers_.begin()->first;
        nextTimeout_ = t.first - utils::steadyMilli();
        nextTimeout_ = nextTimeout_ < 0 ? 0 : nextTimeout_;
    }
}

void EventsImp::repeatableTimeout(TimerRepeatable *tr) {
    tr->at += tr->interval;
    tr->timerid = {tr->at, ++timerSeq_};
    timers_[tr->timerid] = [this, tr] { repeatableTimeout(tr); };
    refreshNearest(&tr->timerid);
    tr->cb();
}