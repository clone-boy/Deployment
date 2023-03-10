#include"src/net/conn.h"
#include<signal.h>

int main(int argc, const char *argv[]) {
    Logger::getLogger().setLogLevel(Logger::LINFO);
    EventBase base;
    Signal::signal(SIGINT, [&] { base.exit(); });

    TcpServerPtr echo = TcpServer::startServer(&base, "", 2099);
    exitif(echo == NULL, "start tcp server failed");
    echo->onConnCreate([] {
        TcpConnPtr con(new TcpConn);
        con->onMsg(new LengthCodec, [](const TcpConnPtr &con, Slice msg) {
            info("recv msg: %.*s", (int) msg.size(), msg.data());
            con->sendMsg(msg);
        });
        return con;
    });
    base.loop();
    info("program exited");
}