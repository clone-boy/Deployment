#include"src/net/conn.h"
#include<signal.h>


int main(int argc, const char *argv[]) {
    setloglevel("INFO");
    EventBase base;
    Signal::signal(SIGINT, [&] { base.exit(); });
    TcpClientPtr cli = TcpClient::createConnection(&base, "127.0.0.1", 2099);
    cli->con_->onMsg(new LengthCodec, [](const TcpConnPtr &con, Slice msg) { info("recv msg: %.*s", (int) msg.size(), 
    msg.data()); });
    cli->con_->onState([=](const TcpConnPtr &con) {
        info("onState called state: %d", con->getState());
        if (con->getState() == State::Connected) {
            con->sendMsg("hello");
        }
    });
    base.loop();
    info("program exited");
}