#include"net/conn.h"
#include"net/logging.h"
#include<fcntl.h>
#include<poll.h>

TcpConn::TcpConn(): base_(NULL), channel_(NULL), state_(State::Invalid), connectedTime_(-1) {}

TcpConn::~TcpConn() {
    trace("tcp destroyed %s - %s", local_.addrToString().c_str(), peer_.addrToString().c_str());
    delete channel_;
}

void TcpConn::attach(EventBase *base, int fd, IPv4Addr local, IPv4Addr peer) {
    fatalif(state_ != State::Invalid && state_ != State::Handshaking,
            "you should use a new TcpConn to attach. state: %d", state_);
    base_ = base;
    state_ = State::Handshaking;
    local_ = local;
    peer_ = peer;
    delete channel_;
    channel_ = new Channel(base, fd, kWriteEvent | kReadEvent);
    trace("tcp constructed %s - %s fd: %d", local_.addrToString().c_str(), peer_.addrToString().c_str(), fd);
    TcpConnPtr con = shared_from_this();
    channel_->onRead([=] { con->handleRead(con); });
    channel_->onWrite([=] { con->handleWrite(con); });
}

void TcpConn::handleRead(const TcpConnPtr &con) {
    if (state_ == State::Handshaking && handleHandshake(con)) {
        return;
    }
    while (state_ == State::Connected) {
        input_.makeRoom();
        int rd = 0;
        if (channel_->fd() >= 0) {
            rd = readImp(channel_->fd(), input_.end(), input_.space());
            trace("channel %lld fd %d readed %d bytes", (long long) channel_->id(), channel_->fd(), rd);
        }
        if (rd == -1 && errno == EINTR) {
            continue;
        } else if (rd == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (readcb_ && input_.size()) {
                readcb_(con);
            }
            break;
        } else if (channel_->fd() == -1 || rd == 0 || rd == -1) {
            cleanup(con);
            break;
        } else {  // rd > 0
            input_.addSize(rd);
        }
    }
}

int TcpConn::handleHandshake(const TcpConnPtr &con) {
    fatalif(state_ != Handshaking, "handleHandshaking called when state_=%d", state_);
    struct pollfd pfd;
    pfd.fd = channel_->fd();
    pfd.events = POLLOUT | POLLERR;
    int r = poll(&pfd, 1, 0);
    if (r == 1 && pfd.revents == POLLOUT) {
        state_ = State::Connected;
        if (state_ == State::Connected) {
            channel_->enableReadWrite(true, false);
            connectedTime_ = utils::steadyMilli();
            info("tcp connected %s - %s fd %d", local_.addrToString().c_str(), peer_.addrToString().c_str(), channel_->fd());
            if (statecb_) {
                statecb_(con);
            }
        }
    }else {
        trace("poll fd %d return %d revents %d", channel_->fd(), r, pfd.revents);
        cleanup(con);
        return -1;
    }
    return 0;
}

void TcpConn::handleWrite(const TcpConnPtr &con) {
    if (state_ == State::Handshaking) {
        handleHandshake(con);
    } else if (state_ == State::Connected) {
        ssize_t sended = isend(output_.begin(), output_.size());
        output_.consume(sended);
        if (output_.empty() && channel_->writeEnabled()) {
            channel_->enableWrite(false);
        }
    } else {
        error("handle write unexpected");
    }
}

void TcpConn::onMsg(CodecBase *codec, const MsgCallBack &cb) {
    assert(!readcb_);
    codec_.reset(codec);
    onRead([cb](const TcpConnPtr &con) {
        int r = 1;
        while (r) {
            Slice msg;
            r = con->codec_->tryDecode(con->getInput(), msg);
            if (r < 0) {
                con->channel_->close();
                break;
            } else if (r > 0) {
                trace("a msg decoded. origin len %d msg len %ld", r, msg.size());
                cb(con, msg);
                con->getInput().consume(r);
            }
        }
    });
}

void TcpConn::sendMsg(Slice msg) {
    codec_->encode(msg, getOutput());
    sendOutput();
}

void TcpConn::send(Buffer &buf) {
    if (channel_) {
        if (channel_->writeEnabled()) {  //由于缓冲区已满或对方读缓冲区已满，此时正在监听写事件，暂存需要发的数据
            output_.absorb(buf);
        }
        if (buf.size()) { //如果没有的话，发送
            ssize_t sended = isend(buf.begin(), buf.size());
            buf.consume(sended);
        }
        if (buf.size()) {
            output_.absorb(buf);
            if (!channel_->writeEnabled()) {
                channel_->enableWrite(true);
            }
        }
    } else {
        warn("connection %s - %s closed, but still writing %lu bytes", local_.addrToString().c_str(), peer_.addrToString().c_str(), buf.size());
    }
}

ssize_t TcpConn::isend(const char *buf, size_t len) {
    size_t sended = 0;
    while (len > sended) {
        ssize_t wd = writeImp(channel_->fd(), buf + sended, len - sended);
        trace("channel %lld fd %d write %ld bytes", (long long) channel_->id(), channel_->fd(), wd);
        if (wd > 0) {
            sended += wd;
            continue;
        } else if (wd == -1 && errno == EINTR) {
            continue;
        } else if (wd == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (!channel_->writeEnabled()) {
                channel_->enableWrite(true);
            }
            break;
        } else {
            error("write error: channel %lld fd %d wd %ld %d %s", (long long) channel_->id(), channel_->fd(), wd, errno, strerror(errno));
            break;
        }
    }
    return sended;
}

void TcpConn::cleanup(const TcpConnPtr &con) {
    if (readcb_ && input_.size()) {
        readcb_(con);
    }
    if (state_ == State::Handshaking) {
        state_ = State::Failed;
    } else {
        state_ = State::Closed;
    }
    trace("tcp closing %s - %s fd %d %d", local_.addrToString().c_str(), peer_.addrToString().c_str(), channel_ ? channel_->fd() : -1, errno);
    getBase()->cancel(timeoutId_);
    if (statecb_) {
        statecb_(con);
    }
    // channel may have hold TcpConnPtr, set channel_ to NULL before delete
    readcb_ = writablecb_ = statecb_ = nullptr;
    Channel *ch = channel_;
    channel_ = NULL;
    delete ch;
}



TcpClient::TcpClient(): base_(NULL), connectTimeout_(0), createcb_([] {return TcpConnPtr(new TcpConn); }) {}

TcpClient::~TcpClient() {
    trace("tcp client shut down %s - %s", local_.addrToString().c_str(), peer_.addrToString().c_str());
}

TcpClientPtr TcpClient::createConnection(EventBase *base, const std::string &host, unsigned short port, int timeout, const std::string &localip) {
    TcpClientPtr p(new TcpClient);
    int r = p->connect(base, host, port, timeout, localip);
    if (r) {
        error("connect to %s:%d failed %d %s", host.c_str(), port, errno, strerror(errno));
    }
    return r == 0 ? p : NULL;
}

int TcpClient::connect(EventBase *base, const std::string &host, unsigned short port, int timeout, const std::string &localip) {
    base_ = base;
    connectTimeout_ = timeout;
    peer_ = IPv4Addr(host, port);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    fatalif(fd < 0, "socket failed %d %s", errno, strerror(errno));
    net::setNonBlock(fd);
    int t = net::addFdFlag(fd, FD_CLOEXEC);
    fatalif(t, "addFdFlag FD_CLOEXEC failed %d %s", t, strerror(t));
    int r = 0;
    if (localip.size()) { //使fd与设置的ip绑定
        IPv4Addr addr(localip, 0); //0为动态绑定端口
        r = ::bind(fd, (struct sockaddr *) &addr.getAddr(), sizeof(struct sockaddr));
        if(r != 0){
            error("bind to %s failed error %d %s", addr.addrToString().c_str(), errno, strerror(errno));
        }
    }
    r = ::connect(fd, (sockaddr *) &peer_.getAddr(), sizeof(sockaddr_in));
    if (r != 0 && errno != EINPROGRESS) {
        error("connect to %s error %d %s", peer_.addrToString().c_str(), errno, strerror(errno));
        return errno;
    }
    sockaddr_in local;
    socklen_t alen = sizeof(local);
    if (r == 0) {
        r = getsockname(fd, (sockaddr *) &local, &alen);
        if (r < 0) {
            error("getsockname failed %d %s", errno, strerror(errno));
        }
    }
    if(r == 0){
        local_ = IPv4Addr(local);
    }

    con_ = createcb_();
    con_->state_ = State::Handshaking;

    if (statecb_) {
        con_->onState(statecb_);
    }
    if (readcb_) {
        con_->onRead(readcb_);
    }
    if (msgcb_) {
        con_->onMsg(codec_->clone(), msgcb_);
    }

    con_->attach(base_, fd, local_, peer_);

    if (timeout) {
        con_->timeoutId_ = base_->runAfter(timeout, [this] {
            if (con_->getState() == Handshaking) {
                con_->channel_->close();
            }
        });
    }
    return 0;
}

void TcpClient::reconnect(long long interval) {
    base_->imp_->reconnectConns_.insert(con_);
    interval = interval > 0 ? interval : 0;
    info("It will be reconnected after %lld ms", interval);
    base_->runAfter(interval, [this]() {
        base_->imp_->reconnectConns_.erase(con_);
        connect(base_, peer_.ipToString(), peer_.port(), connectTimeout_, local_.ipToString());
    });
}




TcpServer::TcpServer():base_(NULL), listen_channel_(NULL), createcb_([] { return TcpConnPtr(new TcpConn); }){}

TcpServer::~TcpServer(){
    delete listen_channel_;
}

TcpServerPtr TcpServer::startServer(EventBase *bases, const std::string &host, unsigned short port, bool reusePort) {
    TcpServerPtr p(new TcpServer());
    int r = p->bind(bases, host, port, reusePort);
    if (r) {
        error("bind to %s:%d failed %d %s", host.c_str(), port, errno, strerror(errno));
    }
    return r == 0 ? p : NULL;
}

int TcpServer::bind(EventBase *base, const std::string &host, unsigned short port, bool reusePort) {
    base_ = base->allocBase();
    addr_ = IPv4Addr(host, port);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int r = net::setReuseAddr(fd);
    fatalif(r, "set socket reuse option failed");
    r = net::setReusePort(fd, reusePort);
    fatalif(r, "set socket reuse port option failed");
    r = net::addFdFlag(fd, FD_CLOEXEC);
    fatalif(r, "addFdFlag FD_CLOEXEC failed");
    r = ::bind(fd, (struct sockaddr *) &addr_.getAddr(), sizeof(struct sockaddr));
    if (r) {
        close(fd);
        error("bind to %s failed %d %s", addr_.addrToString().c_str(), errno, strerror(errno));
        return errno;
    }
    r = listen(fd, 20);
    fatalif(r, "listen failed %d %s", errno, strerror(errno));
    info("fd %d listening at %s", fd, addr_.addrToString().c_str());
    listen_channel_ = new Channel(base_, fd, kReadEvent);
    listen_channel_->onRead([this] { handleAccept(); }); //将接收利用channel接管给base_
    return 0;
}

void TcpServer::handleAccept() {
    struct sockaddr_in raddr;
    socklen_t rsz = sizeof(raddr);
    int lfd = listen_channel_->fd();
    int cfd;
    while (lfd >= 0 && (cfd = accept(lfd, (struct sockaddr *) &raddr, &rsz)) >= 0) {
        sockaddr_in peer, local;
        socklen_t alen = sizeof(peer);
        int r = getpeername(cfd, (sockaddr *) &peer, &alen);
        if (r < 0) {
            error("get peer name failed %d %s", errno, strerror(errno));
            continue;
        }
        r = getsockname(cfd, (sockaddr *) &local, &alen);
        if (r < 0) {
            error("getsockname failed %d %s", errno, strerror(errno));
            continue;
        }
        r = net::addFdFlag(cfd, FD_CLOEXEC);
        fatalif(r, "addFdFlag FD_CLOEXEC failed");
        EventBase *b = base_->allocBase();

        TcpConnPtr con = createcb_();
        con->state_ = State::Handshaking;
        
        if (readcb_) {
            con->onRead(readcb_);
        }
        if (msgcb_) {
            con->onMsg(codec_->clone(), msgcb_);
        }

        con->attach(b, cfd, local, peer);
        
    }
    if (lfd >= 0 && errno != EAGAIN && errno != EINTR) {
        warn("accept return %d  %d %s", cfd, errno, strerror(errno));
    }
}

HSHAPtr HSHA::startServer(EventBase *base, const std::string &host, unsigned short port, int threads) {
    HSHAPtr p = HSHAPtr(new HSHA(threads));
    p->server_ = TcpServer::startServer(base, host, port);
    return p->server_ ? p : NULL;
}

void HSHA::onMsg(CodecBase *codec, const RetMsgCallBack &cb) {
    server_->onConnMsg(codec, [this, cb](const TcpConnPtr &con, Slice msg) {
        std::string input = msg;
        threadPool_.addTask([=] {
            std::string output = cb(con, input);
            server_->getBase()->safeCall([=] {
                if (output.size())
                    con->sendMsg(output);
            });
        });
    });
}