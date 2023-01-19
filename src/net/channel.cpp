#include"net/channel.h"
#include"net/logging.h"
#include"string.h"
#include<atomic>
#include"net/net.h"

Epoller::Epoller()
{   
    lastActive_=-1;
    epfd_ = epoll_create1(EPOLL_CLOEXEC);
    fatalif(epfd_<0, "epoll create error %d %s", errno, strerror(errno));
    info("epoll %d created", epfd_);
}

Epoller::~Epoller()
{
    info("destroying epoller %d", epfd_);
    while (liveChannels_.size()) {
        (*liveChannels_.begin())->close();
    }
    ::close(epfd_);
    info("poller %d destroyed", epfd_);
}

void Epoller::addChannel(Channel *ch){
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.events = ch->events();
    ev.data.ptr = ch;
    trace("adding channel %lld fd %d events %d epoll %d", (long long) ch->id(), ch->fd(), ev.events, epfd_);
    int r = epoll_ctl(epfd_, EPOLL_CTL_ADD, ch->fd(), &ev);
    fatalif(r, "epoll_ctl add failed %d %s", errno, strerror(errno));
    liveChannels_.insert(ch);
}

void Epoller::updateChannel(Channel *ch){
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.events = ch->events();
    ev.data.ptr = ch;
    trace("modifying channel %lld fd %d events read %d write %d epoll %d", (long long) ch->id(), ch->fd(), ev.events & kReadEvent, ev.events & kWriteEvent, epfd_);
    int r = epoll_ctl(epfd_, EPOLL_CTL_MOD, ch->fd(), &ev);
    fatalif(r, "epoll_ctl mod failed %d %s", errno, strerror(errno));
}

void Epoller::removeChannel(Channel *ch){
    trace("deleting channel %lld fd %d epoll %d", (long long) ch->id(), ch->fd(), epfd_);
    liveChannels_.erase(ch);
    for (int i = lastActive_; i >= 0; i--) {
        if (ch == activeEvs_[i].data.ptr) {
            activeEvs_[i].data.ptr = NULL;
            break;
        }
    }
}

void Epoller::loop_once(int waitMs){
    int64_t ticks = utils::steadyMilli();
    lastActive_ = epoll_wait(epfd_, activeEvs_, kMaxEvents, waitMs);
    int64_t used = utils::steadyMilli() - ticks;
    trace("epoll wait %d return %d errno %d used %lld millsecond", waitMs, lastActive_, errno, (long long) used);
    fatalif(lastActive_ == -1 && errno != EINTR, "epoll return error %d %s", errno, strerror(errno));//EINTR被信号中断
    while (--lastActive_ >= 0) {
        int i = lastActive_;
        Channel *ch = (Channel *) activeEvs_[i].data.ptr;
        int events = activeEvs_[i].events;
        if (ch) {
            if (events & (kReadEvent | EPOLLERR)) {
                trace("channel %lld fd %d handle read", (long long) ch->id(), ch->fd());
                ch->handleRead(); 
            } else if (events & kWriteEvent) {
                trace("channel %lld fd %d handle write", (long long) ch->id(), ch->fd());
                ch->handleWrite();
            } else {
                fatal("unexpected epoller events");
            }
        }
    }
}

Epoller * createEpoller(){
    return new Epoller();
}


Channel::Channel(EventBase* base, int fd, int events): eventbase_(base),fd_(fd),events_(events){
    fatalif(net::setNonBlock(fd_) < 0, "channel set non block failed");
    static std::atomic<int64_t> id(0);
    id_ = ++id;
    epoller_ = eventbase_->imp_->epoller_;
    epoller_->addChannel(this);
}

Channel::~Channel(){
    close();
}

void Channel::close(){
    if (fd_ >= 0) {
        trace("close channel %lld fd %d", (long long) id_, fd_);
        epoller_->removeChannel(this);
        ::close(fd_);
        fd_ = -1;
        handleRead();
    }
}

void Channel::enableRead(bool enable) {
    if (enable) {
        events_ |= kReadEvent;
    } else {
        events_ &= ~kReadEvent;
    }
    epoller_->updateChannel(this);
}

void Channel::enableWrite(bool enable) {
    if (enable) {
        events_ |= kWriteEvent;
    } else {
        events_ &= ~kWriteEvent;
    }
    epoller_->updateChannel(this);
}

void Channel::enableReadWrite(bool readable, bool writable) {
    if (readable) {
        events_ |= kReadEvent;
    } else {
        events_ &= ~kReadEvent;
    }
    if (writable) {
        events_ |= kWriteEvent;
    } else {
        events_ &= ~kWriteEvent;
    }
    epoller_->updateChannel(this);
}

bool Channel::readEnabled() {
    return events_ & kReadEvent;
}

bool Channel::writeEnabled() {
    return events_ & kWriteEvent;
}
