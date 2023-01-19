#include"net/buffer.h"

Slice Slice::eatWord() {
    const char *b = pb_;
    while (b < pe_ && isspace(*b)) {
        b++;
    }
    const char *e = b;
    while (e < pe_ && !isspace(*e)) {
        e++;
    }
    pb_ = e;
    return Slice(b, e - b);
}

Slice Slice::eatLine() {
    const char *p = pb_;
    while (pb_ < pe_ && *pb_ != '\n' && *pb_ != '\r') {
        pb_++;
    }
    return Slice(p, pb_ - p);
}

Slice &Slice::trimSpace() {
    while (pb_ < pe_ && isspace(*pb_))
        pb_++;
    while (pb_ < pe_ && isspace(pe_[-1]))
        pe_--;
    return *this;
}


int Slice::compare(const Slice &b) const {
    size_t sz = size(), bsz = b.size();
    const int min_len = (sz < bsz) ? sz : bsz;
    int r = memcmp(pb_, b.pb_, min_len);
    if (r == 0) {
        if (sz < bsz)
            r = -1;
        else if (sz > bsz)
            r = +1;
    }
    return r;
}

std::vector<Slice> Slice::split(char ch) const {
    std::vector<Slice> r;
    const char *pb = pb_;
    for (const char *p = pb_; p < pe_; p++) {
        if (*p == ch) {
            r.push_back(Slice(pb, p));
            pb = p + 1;
        }
    }
    if (pe_ != pb_)
        r.push_back(Slice(pb, pe_));
    return r;
}


char *Buffer::makeRoom(size_t len) {
    if (e_ + len <= cap_) {
    } else if (size() + len < cap_ / 2) {
        moveHead();
    } else {
        expand(len);
    }
    return end();
}

void Buffer::expand(size_t len) {
    size_t ncap = std::max(exp_, std::max(2 * cap_, size() + len));
    char *p = new char[ncap];
    std::copy(begin(), end(), p);
    e_ -= b_;
    b_ = 0;
    delete[] buf_;
    buf_ = p;
    cap_ = ncap;
}

void Buffer::copyFrom(const Buffer &b) {
    memcpy(this, &b, sizeof(b));
    if (b.buf_) {
        buf_ = new char[cap_];
        memcpy(data(), b.begin(), b.size());
    }
}

Buffer &Buffer::absorb(Buffer &buf) {
    if (&buf != this) {
        if (size() == 0) {
            char b[sizeof(buf)];
            memcpy(b, this, sizeof(b));
            memcpy(this, &buf, sizeof(b));
            memcpy(&buf, b, sizeof(b));
            std::swap(exp_, buf.exp_);  // keep the origin exp_
        } else {
            append(buf.begin(), buf.size());
            buf.clear();
        }
    }
    return *this;
}