#ifndef DEF_H
#define DEF_H

#include<functional>
#include"net/buffer.h"

typedef std::function<void()> Task;
typedef std::pair<int64_t, int64_t> TimerId; //时间戳，第一个值为时间，第二个值为序列值

class Channel;
class Epoller;
class TcpConn;
class TcpClient;
class TcpServer;
class HSHA;
class EventBase;
class EventsImp;

typedef std::shared_ptr<TcpConn> TcpConnPtr;
typedef std::shared_ptr<TcpClient> TcpClientPtr;
typedef std::shared_ptr<TcpServer> TcpServerPtr;
typedef std::shared_ptr<HSHA> HSHAPtr;
typedef std::function<void(const TcpConnPtr &)> TcpCallBack;
typedef std::function<void(const TcpConnPtr &, Slice)> MsgCallBack;
typedef std::function<std::string(const TcpConnPtr &, const std::string &)> RetMsgCallBack;

#endif