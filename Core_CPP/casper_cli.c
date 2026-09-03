/* casper_cli.c — Casper sovereign search agent CLI. C11. */
#include "casper_rag.h"
#include "rule_parser.h"
#include "proof_generator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

static void json_str(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p=(const unsigned char *)s; *p; ++p) {
            switch (*p) {
                case '"': fputs("\\\"",fp); break;
                case '\\': fputs("\\\\",fp); break;
                case '\n': fputs("\\n",fp); break;
                case '\r': fputs("\\r",fp); break;
                case '\t': fputs("\\t",fp); break;
                default: if (*p < 0x20u) fprintf(fp,"\\u%04x",*p); else fputc(*p,fp);
            }
        }
    }
    fputc('"', fp);
}

/*
 * Backend selection.
 *
 * This used to be RAG_BACKEND_DDG, hardcoded at the single call site, so
 * every invocation scraped html.duckduckgo.com with no way to switch. Set
 * CASPER_BACKEND=searxng (or bing) to point somewhere else without a rebuild.
 */
static RagBackend pick_backend(void) {
    const char *name = getenv("CASPER_BACKEND");
    if (!name || !name[0]) return RAG_BACKEND_DDG;
    if (!strcmp(name, "searxng")) return RAG_BACKEND_SEARXNG;
    if (!strcmp(name, "bing"))    return RAG_BACKEND_BING;
    if (!strcmp(name, "ddg"))     return RAG_BACKEND_DDG;
    fprintf(stderr, "[casper] unknown CASPER_BACKEND=%s (ddg|bing|searxng), using ddg\n", name);
    return RAG_BACKEND_DDG;
}

static int cmd_verify(const char *proof_path) {
    FILE *fp=fopen(proof_path,"r");
    if(!fp){fprintf(stderr,"[casper] cannot open proof file: %s\n",proof_path);return 3;}
    char line[4096], prompt[1024]={0}, output[1024]={0};
    while(fgets(line,sizeof(line),fp)){
        if(!strncmp(line,"prompt: ",8)){size_t l=strlen(line+8);if(l&&line[8+l-1]=='\n')line[8+l-1]='\0';strncpy(prompt,line+8,sizeof(prompt)-1);prompt[sizeof(prompt)-1]='\0';}
        else if(!strncmp(line,"output: ",8)){size_t l=strlen(line+8);if(l&&line[8+l-1]=='\n')line[8+l-1]='\0';strncpy(output,line+8,sizeof(output)-1);output[sizeof(output)-1]='\0';}
    }
    fclose(fp);
    bool ok=niyah_proof_verify(proof_path,prompt,output,NULL);
    printf("{\"proof_path\":");json_str(stdout,proof_path);printf(",\"valid\":%s}\n",ok?"true":"false");
    return ok?0:1;
}

static void build_answer(const RagCtx *ctx,char *out,size_t max){
    if(!ctx||!out||!max)return;
    if(ctx->n_results<=0){snprintf(out,max,"No sources found for this query.");return;}
    const RagResult *top=&ctx->results[0];
    snprintf(out,max,"Source: %s\n\n%s",top->title[0]?top->title:top->url,top->snippet[0]?top->snippet:"(no snippet)");
}

int main(int argc,char **argv){
    if(argc<2){fprintf(stderr,"usage: %s <query> [rules.nrule] | --verify <proof>\n",argv[0]);return 3;}
    if(!strcmp(argv[1],"--verify")){if(argc<3)return 3;return cmd_verify(argv[2]);}

    const char *query=argv[1];
    const char *rules_path=argc>=3?argv[2]:NULL;
    RagCtx *ctx=casper_rag_query(query,pick_backend(),rules_path);
    if(!ctx){printf("{\"error\":\"rag allocation failure\"}\n");return 2;}
    if(ctx->n_results<=0){
        /* ASCII only: an em-dash here is what produced the mojibake. */
        printf("{\"query\":");json_str(stdout,query);printf(",\"error\":\"no results - offline or no match\",\"confidence\":0.0}\n");
        casper_rag_free(ctx); return 2;
    }

    NiyahRuleKB *kb=NULL;
    if(rules_path){
        kb=niyah_rule_load(rules_path);
        if(!kb){fprintf(stderr,"[casper] failed to load rules: %s\n",rules_path);casper_rag_free(ctx);return 3;}
    }

    char answer[2048];
    build_answer(ctx,answer,sizeof(answer));
    const char *violation=NULL;
    bool rejected=false;
    if(kb){
        violation=niyah_rule_check(kb,query,answer);
        if(violation){
            if(!strcmp(violation,"REJECTED")){rejected=true;snprintf(answer,sizeof(answer),"[Output rejected by symbolic rules]");}
            else {strncpy(answer,violation,sizeof(answer)-1);answer[sizeof(answer)-1]='\0';}
        }
    }

    uint8_t proof_bytes[32];
    niyah_proof_generate(query,answer,rules_path,proof_bytes);
    char proof_hex[65];niyah_hash_to_hex(proof_bytes,proof_hex);
    char proof_path[256];
    int pn=snprintf(proof_path,sizeof(proof_path),"casper_%.8s.proof",proof_hex);
    if(pn<0 || (size_t)pn>=sizeof(proof_path)){if(kb)niyah_rule_free(kb);casper_rag_free(ctx);return 3;}
    if(niyah_proof_save(proof_path,proof_bytes,query,answer,rules_path)!=0){
        fprintf(stderr,"[casper] proof write failed: %s\n",proof_path);
        if(kb)niyah_rule_free(kb);casper_rag_free(ctx);return 1;
    }

    printf("{\n  \"query\":");json_str(stdout,query);printf(",\n  \"answer\":");json_str(stdout,answer);printf(",\n");
    printf("  \"confidence\":%.3f,\n  \"elapsed_ms\":%u,\n  \"violated\":%s,\n  \"rejected\":%s,\n",
           (double)ctx->confidence,ctx->elapsed_ms,violation?"true":"false",rejected?"true":"false");
    printf("  \"proof\":\"%s\",\n  \"proof_file\":",proof_hex);json_str(stdout,proof_path);printf(",\n  \"n_sources\":%d,\n  \"sources\":[\n",ctx->n_results);
    for(int i=0;i<ctx->n_results;++i){
        const RagResult *r=&ctx->results[i];char src_hex[65];niyah_hash_to_hex(r->sha256,src_hex);
        printf("    {\"n\":%d,\"score\":%.3f,\"sha256\":\"%s\",\"title\":",i+1,(double)r->score,src_hex);json_str(stdout,r->title);printf(",\"url\":");json_str(stdout,r->url);printf(",\"snippet\":");json_str(stdout,r->snippet);printf("}%s\n",i+1<ctx->n_results?",":"");
    }
    printf("  ]\n}\n");

    if(kb)niyah_rule_free(kb);
    casper_rag_free(ctx);
    return violation?1:0;
}
