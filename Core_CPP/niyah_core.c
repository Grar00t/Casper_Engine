/* niyah_core.c — NIYAH Inference Engine v3.0 */

#define _GNU_SOURCE
#include "niyah_core.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>

#if defined(_MSC_VER)
#include <windows.h>
static double niyah_monotonic_ms(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER now;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
static double niyah_monotonic_ms(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0)
        return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1000000.0;
}
#endif

#if defined(__AVX2__) && defined(__FMA__)
#  include <immintrin.h>
#  define SIMD_AVX2 1
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
#  define SIMD_NEON 1
#endif

typedef char _cfg_size_check[(sizeof(NiyahConfig) == 64) ? 1 : -1];

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "[niyah] OOM: %zu bytes\n", n); abort(); }
    return p;
}
static void *xcalloc(size_t n, size_t sz) {
    if (sz != 0u && n > SIZE_MAX / sz) {
        fprintf(stderr, "[niyah] allocation overflow: %zu x %zu\n", n, sz);
        abort();
    }
    void *p = calloc(n, sz);
    if (!p) { fprintf(stderr, "[niyah] OOM: %zu x %zu\n", n, sz); abort(); }
    return p;
}

const char *niyah_simd_name(void) {
#if defined(SIMD_AVX2)
    return "AVX2+FMA";
#elif defined(SIMD_NEON)
    return "NEON";
#else
    return "Scalar";
#endif
}

static size_t checked_add(size_t a, size_t b) {
    if (b > SIZE_MAX - a) { fprintf(stderr, "[niyah] size overflow\n"); abort(); }
    return a + b;
}
static size_t checked_mul(size_t a, size_t b) {
    if (a != 0u && b > SIZE_MAX / a) { fprintf(stderr, "[niyah] size overflow\n"); abort(); }
    return a * b;
}

static void matvec(float * restrict y,const float * restrict A,const float * restrict x,size_t R,size_t C) {
#if defined(SIMD_AVX2)
    for (size_t r=0;r<R;r++) { const float *row=A+r*C; __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(); size_t c=0;
        for(;c+15<C;c+=16){a0=_mm256_fmadd_ps(_mm256_loadu_ps(row+c),_mm256_loadu_ps(x+c),a0);a1=_mm256_fmadd_ps(_mm256_loadu_ps(row+c+8),_mm256_loadu_ps(x+c+8),a1);} __m256 acc=_mm256_add_ps(a0,a1);__m128 lo=_mm256_castps256_ps128(acc),hi=_mm256_extractf128_ps(acc,1),s4=_mm_add_ps(lo,hi),s2=_mm_add_ps(s4,_mm_movehdup_ps(s4));float dot=_mm_cvtss_f32(_mm_add_ss(s2,_mm_movehl_ps(s2,s2)));for(;c<C;c++)dot+=row[c]*x[c];y[r]=dot; }
#elif defined(SIMD_NEON)
    for (size_t r=0;r<R;r++){const float*row=A+r*C;float32x4_t a0=vdupq_n_f32(0.f),a1=vdupq_n_f32(0.f);size_t c=0;for(;c+7<C;c+=8){a0=vfmaq_f32(a0,vld1q_f32(row+c),vld1q_f32(x+c));a1=vfmaq_f32(a1,vld1q_f32(row+c+4),vld1q_f32(x+c+4));}float32x4_t acc=vaddq_f32(a0,a1);float32x2_t lo2=vadd_f32(vget_low_f32(acc),vget_high_f32(acc));float dot=vget_lane_f32(vpadd_f32(lo2,lo2),0);for(;c<C;c++)dot+=row[c]*x[c];y[r]=dot;}
#else
    for (size_t r=0;r<R;r++){const float*row=A+r*C;float s0=0.f,s1=0.f,s2=0.f,s3=0.f;size_t c=0;for(;c+3<C;c+=4){s0+=row[c]*x[c];s1+=row[c+1]*x[c+1];s2+=row[c+2]*x[c+2];s3+=row[c+3]*x[c+3];}float dot=s0+s1+s2+s3;for(;c<C;c++)dot+=row[c]*x[c];y[r]=dot;}
#endif
}

static float dot_f32(const float * restrict a,const float * restrict b,size_t n){
#if defined(SIMD_AVX2)
    __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps();size_t i=0;for(;i+15<n;i+=16){a0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),a0);a1=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8),a1);}__m256 acc=_mm256_add_ps(a0,a1);__m128 lo=_mm256_castps256_ps128(acc),hi=_mm256_extractf128_ps(acc,1),s4=_mm_add_ps(lo,hi),s2=_mm_add_ps(s4,_mm_movehdup_ps(s4));float d=_mm_cvtss_f32(_mm_add_ss(s2,_mm_movehl_ps(s2,s2)));for(;i<n;i++)d+=a[i]*b[i];return d;
#elif defined(SIMD_NEON)
    float32x4_t a0=vdupq_n_f32(0.f),a1=vdupq_n_f32(0.f);size_t i=0;for(;i+7<n;i+=8){a0=vfmaq_f32(a0,vld1q_f32(a+i),vld1q_f32(b+i));a1=vfmaq_f32(a1,vld1q_f32(a+i+4),vld1q_f32(b+i+4));}float32x4_t acc=vaddq_f32(a0,a1);float32x2_t lo2=vadd_f32(vget_low_f32(acc),vget_high_f32(acc));float d=vget_lane_f32(vpadd_f32(lo2,lo2),0);for(;i<n;i++)d+=a[i]*b[i];return d;
#else
    float d=0.f;for(size_t i=0;i<n;i++)d+=a[i]*b[i];return d;
#endif
}

static void rmsnorm(float * restrict out,const float * restrict x,const float * restrict w,size_t n,float eps){
    if(n==0u) return; float ss=dot_f32(x,x,n);float scale=1.0f/sqrtf(ss/(float)n+eps);for(size_t k=0;k<n;k++)out[k]=x[k]*scale*w[k];
}
static inline float silu(float v){return v/(1.0f+expf(-v));}
static void rope(float*qk,uint32_t pos,uint32_t head_dim,float theta){for(uint32_t i=0;i+1<head_dim;i+=2){float angle=(float)pos/powf(theta,(float)i/(float)head_dim);float c=cosf(angle),s=sinf(angle),v0=qk[i],v1=qk[i+1];qk[i]=v0*c-v1*s;qk[i+1]=v0*s+v1*c;}}

static size_t weight_count(const NiyahConfig*c){size_t d=c->embed_dim,kd=checked_mul(c->n_kv_heads,d/c->n_heads),f=checked_mul(d,c->ffn_mult),L=c->n_layers,v=c->vocab_size;size_t pl=checked_add(checked_add(checked_add(checked_add(checked_mul(d,d),checked_mul(kd,d)),checked_add(checked_mul(kd,d),checked_mul(d,d))),checked_add(checked_mul(f,d),checked_mul(f,d))),checked_add(checked_add(checked_mul(d,f),d),d));return checked_add(checked_mul(L,pl),checked_add(checked_mul(v,d),checked_add(d,checked_mul(v,d))));}
static size_t kv_count(const NiyahConfig*c){size_t hd=c->embed_dim/c->n_heads;return checked_mul(checked_mul(checked_mul(checked_mul(2u,c->n_layers),c->n_kv_heads),c->ctx_len),hd);}
static size_t scratch_count(const NiyahConfig*c){size_t d=c->embed_dim,f=checked_mul(d,c->ffn_mult),a=checked_mul(c->n_heads,c->ctx_len);return checked_add(checked_add(checked_add(checked_mul(3u,d),checked_mul(2u,f)),checked_mul(3u,d)),checked_add(a,c->vocab_size));}

size_t niyah_param_count(const NiyahModel*m){return m?weight_count(&m->cfg):0u;}

NiyahModel*niyah_alloc(const NiyahConfig*cfg){
    if(!cfg||cfg->embed_dim==0u||cfg->n_heads==0u||cfg->n_layers==0u||cfg->n_kv_heads==0u||cfg->n_kv_heads>cfg->n_heads||cfg->vocab_size==0u||cfg->ctx_len==0u||cfg->ffn_mult==0u||cfg->embed_dim%cfg->n_heads!=0u||cfg->ctx_len>NIYAH_MAX_CTX||cfg->vocab_size>NIYAH_MAX_VOCAB) return NULL;
    NiyahModel*m=xcalloc(1u,sizeof(*m));m->cfg=*cfg;m->head_dim=cfg->embed_dim/cfg->n_heads;m->kv_dim=cfg->n_kv_heads*m->head_dim;m->ffn_dim=cfg->embed_dim*cfg->ffn_mult;size_t nw=weight_count(cfg),nkv=kv_count(cfg),nsc=scratch_count(cfg),nl=cfg->n_layers;size_t float_count=checked_add(checked_add(nw,nkv),nsc);size_t float_bytes=checked_mul(float_count,sizeof(float));size_t align_pad=(float_bytes%_Alignof(NiyahLayer))?(_Alignof(NiyahLayer)-(float_bytes%_Alignof(NiyahLayer))):0u;m->_pool_bytes=checked_add(float_bytes,checked_add(align_pad,checked_mul(nl,sizeof(NiyahLayer))));m->_pool=xcalloc(1u,m->_pool_bytes);float*fp=(float*)m->_pool;NiyahLayer*lp=(NiyahLayer*)((char*)m->_pool+float_bytes+align_pad);m->layers=lp;size_t d=cfg->embed_dim,kd=m->kv_dim,f=m->ffn_dim;float*p=fp;for(uint32_t l=0;l<nl;l++){m->layers[l].wq=p;p+=checked_mul(d,d);m->layers[l].wk=p;p+=checked_mul(kd,d);m->layers[l].wv=p;p+=checked_mul(kd,d);m->layers[l].wo=p;p+=checked_mul(d,d);m->layers[l].w_gate=p;p+=checked_mul(f,d);m->layers[l].w_up=p;p+=checked_mul(f,d);m->layers[l].w_down=p;p+=checked_mul(d,f);m->layers[l].rms_att=p;p+=d;m->layers[l].rms_ffn=p;p+=d;}m->token_embed=p;p+=checked_mul(cfg->vocab_size,d);m->rms_final=p;p+=d;m->lm_head=p;p+=checked_mul(cfg->vocab_size,d);m->kv_k=p;p+=nkv/2u;m->kv_v=p;p+=nkv/2u;m->scratch=p;m->_logits=p+3u*d+2u*f+3u*d+(size_t)cfg->n_heads*cfg->ctx_len;assert((size_t)(p-fp)==nw+nkv);return m;}

void niyah_free(NiyahModel*m){if(!m)return;free(m->_pool);free(m);}

int niyah_save(const NiyahModel*m,const char*path){if(!m||!path)return -1;FILE*fp=fopen(path,"wb");if(!fp)return -1;NiyahConfig hdr=m->cfg;hdr.magic=NIYAH_MAGIC;hdr.version=NIYAH_VER;size_t nw=weight_count(&m->cfg);int ok=(fwrite(&hdr,sizeof hdr,1,fp)==1)&&(fwrite(m->_pool,sizeof(float),nw,fp)==nw);if(fclose(fp)!=0)ok=0;return ok?0:-1;}

int niyah_load(NiyahModel**out,const char*path){if(!out||!path)return -1;*out=NULL;FILE*fp=fopen(path,"rb");if(!fp)return -1;NiyahConfig cfg={0};if(fread(&cfg,sizeof cfg,1,fp)!=1){fclose(fp);return -1;}if(cfg.magic!=NIYAH_MAGIC){fclose(fp);return -1;}if(cfg.version!=NIYAH_VER){fclose(fp);return -2;}NiyahModel*m=niyah_alloc(&cfg);if(!m){fclose(fp);return -1;}size_t nw=weight_count(&cfg);if(fread(m->_pool,sizeof(float),nw,fp)!=nw){niyah_free(m);fclose(fp);return -1;}fclose(fp);*out=m;return 0;}

#define SCR_X(m)((m)->scratch)
#define SCR_XB(m)((m)->scratch+(m)->cfg.embed_dim)
#define SCR_XB2(m)((m)->scratch+2u*(m)->cfg.embed_dim)
#define SCR_HB(m)((m)->scratch+3u*(m)->cfg.embed_dim)
#define SCR_HB2(m)((m)->scratch+3u*(m)->cfg.embed_dim+(m)->ffn_dim)
#define SCR_Q(m)((m)->scratch+3u*(m)->cfg.embed_dim+2u*(m)->ffn_dim)
#define SCR_K(m)((m)->scratch+4u*(m)->cfg.embed_dim+2u*(m)->ffn_dim)
#define SCR_V(m)((m)->scratch+4u*(m)->cfg.embed_dim+2u*(m)->ffn_dim+(m)->kv_dim)
#define SCR_ATT(m)((m)->scratch+4u*(m)->cfg.embed_dim+2u*(m)->ffn_dim+2u*(m)->kv_dim)

float*niyah_forward(NiyahModel*m,uint32_t token,uint32_t pos){if(!m||!m->_pool||token>=m->cfg.vocab_size||pos>=m->cfg.ctx_len)return NULL;const NiyahConfig*c=&m->cfg;uint32_t d=c->embed_dim,hd=m->head_dim,nh=c->n_heads,nkv=c->n_kv_heads,xctx=c->ctx_len;float*x=SCR_X(m),*xb=SCR_XB(m),*xb2=SCR_XB2(m),*hb=SCR_HB(m),*hb2=SCR_HB2(m),*q=SCR_Q(m),*k=SCR_K(m),*v=SCR_V(m),*att=SCR_ATT(m);memcpy(x,m->token_embed+(size_t)token*d,d*sizeof(float));for(uint32_t l=0;l<c->n_layers;l++){const NiyahLayer*lw=&m->layers[l];rmsnorm(xb,x,lw->rms_att,d,c->rms_eps);matvec(q,lw->wq,xb,d,d);matvec(k,lw->wk,xb,m->kv_dim,d);matvec(v,lw->wv,xb,m->kv_dim,d);for(uint32_t h=0;h<nh;h++)rope(q+h*hd,pos,hd,c->rope_theta);for(uint32_t h=0;h<nkv;h++)rope(k+h*hd,pos,hd,c->rope_theta);size_t L=(size_t)nkv*xctx*hd;float*kc=m->kv_k+(size_t)l*L,*vc=m->kv_v+(size_t)l*L;for(uint32_t h=0;h<nkv;h++){float*dk=kc+(size_t)h*xctx*hd+(size_t)pos*hd;float*dv=vc+(size_t)h*xctx*hd+(size_t)pos*hd;memcpy(dk,k+h*hd,hd*sizeof(float));memcpy(dv,v+h*hd,hd*sizeof(float));}memset(att,0,(size_t)nh*xctx*sizeof(float));for(uint32_t h=0;h<nh;h++){uint32_t kvh=(h*nkv)/nh;float*ah=att+(size_t)h*xctx;for(uint32_t t=0;t<=pos;t++){float score=dot_f32(q+h*hd,kc+(size_t)kvh*xctx*hd+(size_t)t*hd,hd)/(sqrtf((float)hd));if(score>80.f)score=80.f;if(score<-80.f)score=-80.f;ah[t]=score;}float maxs=ah[0];for(uint32_t t=1;t<=pos;t++)if(ah[t]>maxs)maxs=ah[t];float den=0.f;for(uint32_t t=0;t<=pos;t++){ah[t]=expf(ah[t]-maxs);den+=ah[t];}den=den>0.f?den:1.f;for(uint32_t t=0;t<=pos;t++)ah[t]/=den;memset(xb2+h*hd,0,hd*sizeof(float));for(uint32_t t=0;t<=pos;t++){const float*vv=vc+(size_t)kvh*xctx*hd+(size_t)t*hd;for(uint32_t j=0;j<hd;j++)xb2[h*hd+j]+=ah[t]*vv[j];}}matvec(xb,lw->wo,xb2,d,d);for(uint32_t i=0;i<d;i++)x[i]+=xb[i];rmsnorm(xb,x,lw->rms_ffn,d,c->rms_eps);matvec(hb,lw->w_gate,xb,m->ffn_dim,d);matvec(hb2,lw->w_up,xb,m->ffn_dim,d);for(uint32_t i=0;i<m->ffn_dim;i++)hb[i]=silu(hb[i])*hb2[i];matvec(xb,lw->w_down,hb,d,m->ffn_dim);for(uint32_t i=0;i<d;i++)x[i]+=xb[i];}rmsnorm(xb,x,m->rms_final,d,c->rms_eps);matvec(m->_logits,m->lm_head,xb,c->vocab_size,d);return m->_logits;}

uint32_t niyah_sample(const float*logits,uint32_t vocab_size,NiyahSampler*s){if(!logits||!s||vocab_size==0u)return 0u;if(!(s->temperature>=0.f)||!isfinite(s->temperature)||!isfinite(s->top_p))return 0u;if(s->temperature<=0.f){uint32_t best=0u;for(uint32_t i=1;i<vocab_size;i++)if(logits[i]>logits[best])best=i;return best;}float mx=logits[0];for(uint32_t i=1;i<vocab_size;i++)if(logits[i]>mx)mx=logits[i];float sm=0.f;for(uint32_t i=0;i<vocab_size;i++){float z=(logits[i]-mx)/s->temperature; if(z<-80.f)z=-80.f;sm+=expf(z);}if(!(sm>0.f)||!isfinite(sm))return 0u;s->seed=s->seed*6364136223846793005ULL+1442695040888963407ULL;float r=(float)((s->seed>>11)&0x0FFFFFFU)/(float)0x0FFFFFFU;float top=(s->top_p>0.f&&s->top_p<1.f)?s->top_p:1.f;float target=r*sm*top;float cum=0.f;for(uint32_t i=0;i<vocab_size;i++){float z=(logits[i]-mx)/s->temperature;if(z<-80.f)z=-80.f;cum+=expf(z);if(cum>=target)return i;}return vocab_size-1u;}

NiyahAdam*niyah_adam_alloc(const NiyahModel*m){if(!m)return NULL;NiyahAdam*opt=xcalloc(1u,sizeof(*opt));opt->n_weights=weight_count(&m->cfg);opt->m=xcalloc(opt->n_weights,sizeof(float));opt->v=xcalloc(opt->n_weights,sizeof(float));opt->lr=3e-4f;opt->beta1=.9f;opt->beta2=.999f;opt->eps=1e-8f;opt->wd=.01f;return opt;}
void niyah_adam_free(NiyahAdam*opt){if(!opt)return;free(opt->m);free(opt->v);free(opt);}

float niyah_train_step(NiyahModel*m,NiyahAdam*opt,const uint32_t*tokens,uint32_t n){if(!m||!opt||!tokens||n<2u)return 0.f;for(uint32_t i=0;i<n;i++)if(tokens[i]>=m->cfg.vocab_size||tokens[i]>=m->cfg.ctx_len)return NAN;size_t nw=weight_count(&m->cfg);float*grad=xcalloc(nw,sizeof(float));float*dL=xmalloc((size_t)m->cfg.vocab_size*sizeof(float));float loss=0.f;uint32_t d=m->cfg.embed_dim;for(uint32_t t=0;t+1<n;t++){const float*logits=niyah_forward(m,tokens[t],t);if(!logits){free(dL);free(grad);return NAN;}uint32_t tgt=tokens[t+1];float mx=logits[0];for(uint32_t i=1;i<m->cfg.vocab_size;i++)if(logits[i]>mx)mx=logits[i];float lse_sum=0.f;for(uint32_t i=0;i<m->cfg.vocab_size;i++){float z=logits[i]-mx;if(z<-80.f)z=-80.f;lse_sum+=expf(z);}if(!(lse_sum>0.f)||!isfinite(lse_sum)){free(dL);free(grad);return NAN;}float lse=logf(lse_sum)+mx;loss+=lse-logits[tgt];for(uint32_t i=0;i<m->cfg.vocab_size;i++)dL[i]=expf(logits[i]-lse);dL[tgt]-=1.f;const float*xb=SCR_XB(m);size_t off=nw-(size_t)m->cfg.vocab_size*d;float*dW=grad+off;for(uint32_t i=0;i<m->cfg.vocab_size;i++){float dl=dL[i];for(uint32_t j=0;j<d;j++)dW[(size_t)i*d+j]+=dl*xb[j];}}
    loss/=(float)(n-1u);opt->step++;float bc1=1.f-powf(opt->beta1,(float)opt->step),bc2=1.f-powf(opt->beta2,(float)opt->step);if(!(bc1>0.f&&bc2>0.f)){free(dL);free(grad);return NAN;}size_t off=nw-(size_t)m->cfg.vocab_size*d;float*W=(float*)m->_pool+off;const float*gW=grad+off;float*mW=opt->m+off,*vW=opt->v+off;size_t cnt=(size_t)m->cfg.vocab_size*d;for(size_t i=0;i<cnt;i++){float g=gW[i]+opt->wd*W[i];mW[i]=opt->beta1*mW[i]+(1.f-opt->beta1)*g;vW[i]=opt->beta2*vW[i]+(1.f-opt->beta2)*g*g;float mh=mW[i]/bc1,vh=vW[i]/bc2;if(!isfinite(mh)||!isfinite(vh)){free(dL);free(grad);return NAN;}W[i]-=opt->lr*mh/(sqrtf(vh)+opt->eps);}free(dL);free(grad);return isfinite(loss)?loss:NAN;}
