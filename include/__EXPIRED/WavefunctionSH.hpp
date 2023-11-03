#ifndef __WAVE_FUNCTION_SH_H__
#define __WAVE_FUNCTION_SH_H__

#include "Wavefunction.hpp"
#include "Util.hpp"

namespace CQP {

class WavefunctionSH
{
public:

    WavefunctionSH() {}

    WavefunctionSH(const WavefunctionSH& shwave)
        : rgrid(shwave.rgrid), rfunc_collection(shwave.rfunc_collection),
        r_appendix(shwave.r_appendix), maxl(shwave.maxl) {}

    WavefunctionSH(WavefunctionSH&& shwave) noexcept
        : rgrid(shwave.rgrid), rfunc_collection(std::move(shwave.rfunc_collection)),
        r_appendix(std::move(shwave.r_appendix)), maxl(shwave.maxl) {}

    WavefunctionSH& operator=(const WavefunctionSH& shwave) {
        rgrid = shwave.rgrid;
        rfunc_collection = shwave.rfunc_collection;
        r_appendix = shwave.r_appendix;
        maxl = shwave.maxl;
        return *this;
    }
    
    WavefunctionSH& operator=(WavefunctionSH&& shwave) noexcept {
        rgrid = shwave.rgrid;
        rfunc_collection = std::move(shwave.rfunc_collection);
        r_appendix = std::move(shwave.r_appendix);
        maxl = shwave.maxl;
        return *this;
    }

    //////////////////////

    WavefunctionSH(const NumericalGrid1D& rgrid) 
        : rgrid(rgrid), maxl(-1) 
        { update_r_appendix(); }

    WavefunctionSH(const NumericalGrid1D& rgrid, int maxl)
        : rgrid(rgrid), maxl(maxl), rfunc_collection((maxl + 1) * (maxl + 1), Wavefunction1D(rgrid))
        { update_r_appendix(); }
    
    ////

    static int getNumberL(int order) 
        { return (int)std::floor(sqrt(order)); }

    static int getNumberM(int order) 
        { int l = getNumberL(order); return -l + (order - l * l); }

    static int getIndex(int l, int m)
        { return l * l + m + l; }

    //////////////////////

    int size() const 
        { return rfunc_collection.size(); }

    Wavefunction1D& getRfunc(int l, int m)
        { assert(l >= 0 && abs(m) <= l); return rfunc_collection[getIndex(l, m)]; }
    
    Wavefunction1D& getRfuncByIndex(int index)
        { return rfunc_collection[index]; }
    
    Wavefunction1D& operator[](int index)
        { return rfunc_collection[index]; }
    
    NumericalGrid1D getRGrid() const
        { return rgrid; }

    WavefunctionSH operator+(const WavefunctionSH& shwave) const ;
    WavefunctionSH operator-(const WavefunctionSH& shwave) const ;
    //WavefunctionSH operator*(const WavefunctionSH& shwave) const ;
    WavefunctionSH operator*(std::complex<double> scaler) const ;
    //WavefunctionSH operator/(const WavefunctionSH& shwave) const ;

    std::complex<double> innerProduct(const WavefunctionSH& shwave) const;

    void normalize();

private:
    void update_r_appendix() {
        MathFuncType<1> rfunc = [](double r){ return 1.0 / (r * r); };
        r_appendix = createWaveByExpression(rgrid, rfunc);
    }

    int maxl;
    NumericalGrid1D rgrid;
    Wavefunction1D r_appendix;
    std::vector< Wavefunction1D > rfunc_collection;
};



WavefunctionSH WavefunctionSH::operator+(const WavefunctionSH& shwave) const
{
    assert(shwave.rgrid == rgrid && size() == shwave.size());
    WavefunctionSH ans(rgrid, maxl);
    for(int i = 0; i < size(); i++) {
        ans.rfunc_collection[i] = rfunc_collection[i] + shwave.rfunc_collection[i];
    }
    return ans;
}

WavefunctionSH WavefunctionSH::operator-(const WavefunctionSH& shwave) const
{
    assert(shwave.rgrid == rgrid && size() == shwave.size());
    WavefunctionSH ans(rgrid, maxl);
    for(int i = 0; i < size(); i++) {
        ans.rfunc_collection[i] = rfunc_collection[i] - shwave.rfunc_collection[i];
    }
    return ans;
}

WavefunctionSH WavefunctionSH::operator*(std::complex<double> scaler) const
{
    WavefunctionSH ans(rgrid, maxl);
    for(int i = 0; i < size(); i++) {
        ans.rfunc_collection[i] = rfunc_collection[i] * scaler;
    }
    return ans;
}

void WavefunctionSH::normalize()
{
    double scaler = 1.0 / sqrt(innerProduct(*this).real());
    for(int i = 0; i < size(); i++) {
        rfunc_collection[i] = rfunc_collection[i] * scaler;
    }
}

std::complex<double> WavefunctionSH::innerProduct(const WavefunctionSH & shwave) const
{
    assert(shwave.rgrid == rgrid && size() == shwave.size());
    std::complex<double> ans = 0;
    for(int i = 0; i < size(); i++){
        ans += (rfunc_collection[i]).innerProduct(shwave.rfunc_collection[i]); 
    }
    return ans;
}


}



#endif //__WAVE_FUNCTION_SH_H__