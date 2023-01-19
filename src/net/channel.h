#ifndef CHANNEL_H
#define CHANNEL_H
#include"net/utils.h"
#include<functional>
#include"net/threads.h"
#include<sys/epoll.h>
#include<set>
#include"net/event_base.h"
#include<errno.h>
#include<stdio.h>
#include"net/def.h"


const int kMaxEvents = 2000;
const int kReadEvent = EPOLLIN;
const int kWriteEvent = EPOLLOUT;

class Epoller{
// epoll树
private:
    int lastActive_;
    int epfd_; // 管理各个epoll实例
    std::set<Channel *> liveChannels_;
    struct epoll_event activeEvs_[kMaxEvents]; //容量
public:
    Epoller();
    ~Epoller();
    void addChannel(Channel *ch);
    void updateChannel(Channel *ch);
    void removeChannel(Channel *ch);
    void loop_once(int waitMs); //循环一次处理fd事件
};


Epoller * createEpoller();

class Channel: private noncopyable{
//管理某个fd
private:
    int fd_;
    int64_t id_; //fd数量，from 1
    short events_; //fd事件
    Epoller* epoller_; //利用传入的事件派发器的epoller控制该Channel
    EventBase* eventbase_; //事件派发器
    std::function<void()> readcb_, writecb_, errorcb_;
public:
    Channel(EventBase* base, int fd, int events);
    ~Channel();
    //关闭通道
    void close();

    int fd(){ return fd_; }
    int64_t id(){ return id_; }
    short events(){ return events_; }

    //挂接事件处理器
    void onRead(const Task &readcb) { readcb_ = readcb; }
    void onWrite(const Task &writecb) { writecb_ = writecb; }
    void onRead(Task &&readcb) { readcb_ = std::move(readcb); }
    void onWrite(Task &&writecb) { writecb_ = std::move(writecb); }

    //启用读写监听
    void enableRead(bool enable);
    void enableWrite(bool enable);
    void enableReadWrite(bool readable, bool writable);
    bool readEnabled();
    bool writeEnabled();

    //处理读写事件
    void handleRead() { readcb_(); }
    void handleWrite() { writecb_(); }
};

#endif