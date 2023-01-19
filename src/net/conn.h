#ifndef CONN_H
#define CONN_H

#include"net/event_base.h"
#include"net/net.h"
#include<assert.h>
#include"net/buffer.h"
#include"net/def.h"
#include"net/threads.h"

enum State {
    Invalid = 1,
    Handshaking,
    Connected,
    Closed,
    Failed,
};

class TcpConn : public std::enable_shared_from_this<TcpConn>, private noncopyable{
public:
    // Tcp构造函数，实际可用的连接应当通过createConnection创建
    TcpConn();
    ~TcpConn();
    //可传入连接类型，返回智能指针

    void attach(EventBase *base, int fd, IPv4Addr local, IPv4Addr peer);

    State getState() { return state_; }
    EventBase *getBase() { return base_; }
    Buffer &getInput() { return input_; }
    Buffer &getOutput() { return output_; }

    void handleRead(const TcpConnPtr &con);
    void handleWrite(const TcpConnPtr &con);
    int handleHandshake(const TcpConnPtr &con);

    void onState(const TcpCallBack &cb) { statecb_ = cb; }
    //数据到达时回调
    void onRead(const TcpCallBack &cb) {
        assert(!readcb_);
        readcb_ = cb;
    };
    //消息回调，此回调与onRead回调冲突，只能够调用一个
    // codec所有权交给onMsg
    void onMsg(CodecBase *codec, const MsgCallBack &cb);

    //发送消息
    void sendMsg(Slice msg);
    void sendOutput() { send(output_); }
    void send(Buffer &msg);
    ssize_t isend(const char *buf, size_t len);

    int readImp(int fd, void *buf, size_t bytes) { return ::read(fd, buf, bytes); }
    int writeImp(int fd, const void *buf, size_t bytes) { return ::write(fd, buf, bytes); }

    void cleanup(const TcpConnPtr &con);

public:
    EventBase *base_;
    Channel *channel_;
    State state_;
    Buffer input_, output_;
    IPv4Addr local_, peer_;
    int64_t connectedTime_;
    TcpCallBack statecb_, readcb_, writablecb_;
    std::unique_ptr<CodecBase> codec_;
    TimerId timeoutId_;
};



class TcpClient: private noncopyable{
public:
    TcpClient();
    ~TcpClient();
    static TcpClientPtr createConnection(EventBase *base, const std::string &host, unsigned short port, int timeout = 0, const std::string &localip = "");

    int connect(EventBase *base, const std::string &host, unsigned short port, int timeout, const std::string &localip);
    void reconnect(long long interval);

    void onConnCreate(const std::function<TcpConnPtr()> &cb) { createcb_ = cb; }
    void onConnState(const TcpCallBack &cb) { statecb_ = cb; }
    void onConnRead(const TcpCallBack &cb) {
        readcb_ = cb;
        assert(!msgcb_);
    }

    // 消息处理与Read回调冲突，只能调用一个
    void onConnMsg(CodecBase *codec, const MsgCallBack &cb) {
        codec_.reset(codec);
        msgcb_ = cb;
        assert(!readcb_);
    }

public:
    EventBase *base_;
    IPv4Addr local_, peer_;
    TcpConnPtr con_;
    int connectTimeout_;
    TcpCallBack statecb_, readcb_;
    MsgCallBack msgcb_;
    std::function<TcpConnPtr()> createcb_;
    std::unique_ptr<CodecBase> codec_;
};


class TcpServer: private noncopyable{
public:
    TcpServer();
    ~TcpServer();

    static TcpServerPtr startServer(EventBase *base, const std::string &host, unsigned short port, bool   
    reusePort = false);

    int bind(EventBase *base, const std::string &host, unsigned short port, bool reusePort = false);
    void handleAccept();

    EventBase *getBase() { return base_; }

    void onConnCreate(const std::function<TcpConnPtr()> &cb) { createcb_ = cb; }

    void onConnRead(const TcpCallBack &cb) {
        readcb_ = cb;
        assert(!msgcb_);
    }
    // 消息处理与Read回调冲突，只能调用一个
    void onConnMsg(CodecBase *codec, const MsgCallBack &cb) {
        codec_.reset(codec);
        msgcb_ = cb;
        assert(!readcb_);
    }
private:
    EventBase *base_;
    Channel *listen_channel_;
    IPv4Addr addr_;
    TcpCallBack readcb_;
    MsgCallBack msgcb_;
    std::function<TcpConnPtr()> createcb_;
    std::unique_ptr<CodecBase> codec_;
};

class HSHA {
public:
    static HSHAPtr startServer(EventBase *base, const std::string &host, unsigned short port, int threads);
    HSHA(int threads) : threadPool_(threads) {}

    void onMsg(CodecBase *codec, const RetMsgCallBack &cb);
    TcpServerPtr server_;
    Thread_Pool threadPool_;
};


#endif