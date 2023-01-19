#ifndef BUFFER_H
#define BUFFER_H

#include<string>
#include<string.h>
#include<vector>

class Slice {
public:
    Slice() : pb_("") { pe_ = pb_; }
    Slice(const char *b, const char *e) : pb_(b), pe_(e) {}
    Slice(const char *d, size_t n) : pb_(d), pe_(d + n) {}
    Slice(const std::string &s) : pb_(s.data()), pe_(s.data() + s.size()) {}
    Slice(const char *s) : pb_(s), pe_(s + strlen(s)) {}

    const char *data() const { return pb_; }
    const char *begin() const { return pb_; }
    const char *end() const { return pe_; }
    char front() { return *pb_; }
    char back() { return pe_[-1]; }
    size_t size() const { return pe_ - pb_; }
    void resize(size_t sz) { pe_ = pb_ + sz; }
    inline bool empty() const { return pe_ == pb_; }
    void clear() { pe_ = pb_ = ""; }

    // return the eated data
    Slice eatWord();
    Slice eatLine();
    Slice eat(int sz) {
        Slice s(pb_, sz);
        pb_ += sz;
        return s;
    }
    Slice sub(int boff, int eoff = 0) const {
        Slice s(*this);
        s.pb_ += boff;
        s.pe_ += eoff;
        return s;
    }
    Slice &trimSpace();

    inline char operator[](size_t n) const { return pb_[n]; }

    inline bool operator<(const Slice &y) {
        return compare(y) < 0;
    }

    inline bool operator==(const Slice &y) {
        return ((size() == y.size()) && (memcmp(data(), y.data(), size()) == 0));
    }

    inline bool operator!=(const Slice &y) {
        return !(*this == y);
    }

    std::string toString() const { return std::string(pb_, pe_); }
    // Three-way comparison.  Returns value:
    int compare(const Slice &b) const;

    // Return true if "x" is a prefix of "*this"
    bool starts_with(const Slice &x) const { return (size() >= x.size() && memcmp(pb_, x.pb_, x.size()) == 0); }

    bool end_with(const Slice &x) const { return (size() >= x.size() && memcmp(pe_ - x.size(), x.pb_, x.size()) == 0); }
    operator std::string() const { return std::string(pb_, pe_); }
    std::vector<Slice> split(char ch) const;

private:
    const char *pb_;
    const char *pe_;
};

class Buffer {
public:
    Buffer() : buf_(NULL), b_(0), e_(0), cap_(0), exp_(512) {}
    ~Buffer() { delete[] buf_; }
    void clear() {
        delete[] buf_;
        buf_ = NULL;
        cap_ = 0;
        b_ = e_ = 0;
    }
    size_t size() const { return e_ - b_; }
    bool empty() const { return e_ == b_; }
    char *data() const { return buf_ + b_; }
    char *begin() const { return buf_ + b_; }
    char *end() const { return buf_ + e_; }
    char *makeRoom(size_t len);
    void makeRoom() {
        if (space() < exp_)
            expand(0);
    }
    size_t space() const { return cap_ - e_; }
    void addSize(size_t len) { e_ += len; }
    char *allocRoom(size_t len) {
        char *p = makeRoom(len);
        addSize(len);
        return p;
    }
    Buffer &append(const char *p, size_t len) {
        memcpy(allocRoom(len), p, len);
        return *this;
    }
    Buffer &append(Slice slice) { return append(slice.data(), slice.size()); }
    Buffer &append(const char *p) { return append(p, strlen(p)); }
    template <class T>
    Buffer &appendValue(const T &v) {
        append((const char *) &v, sizeof v);
        return *this;
    }
    Buffer &consume(size_t len) {
        b_ += len;
        if (size() == 0)
            clear();
        return *this;
    }
    Buffer &absorb(Buffer &buf);
    void setSuggestSize(size_t sz) { exp_ = sz; }
    Buffer(const Buffer &b) { copyFrom(b); }
    Buffer &operator=(const Buffer &b) {
        if (this == &b)
            return *this;
        delete[] buf_;
        buf_ = NULL;
        copyFrom(b);
        return *this;
    }
    operator Slice() { return Slice(data(), size()); }

private:
    char *buf_;
    size_t b_, e_, cap_, exp_;
    void moveHead() {
        std::copy(begin(), end(), buf_);
        e_ -= b_;
        b_ = 0;
    }
    void expand(size_t len);
    void copyFrom(const Buffer &b);
};


#endif