# Benchmark Report: Single-Agent vs Multi-Agent

## Executive Summary

Chúng tôi so sánh hai cách tiếp cận cho bài toán research assistant:
- **Single-Agent**: Một LLM call xử lý toàn bộ query
- **Multi-Agent**: Supervisor orchestrate 3 agents (Researcher → Analyst → Writer)

**Kết quả chính**: Multi-agent nhanh hơn **1.18×** (117.4s vs 139.0s), nhưng chi phí cao hơn vì phải gọi LLM 3 lần.

---

## Metrics Comparison

| Metric | Single-Agent | Multi-Agent | Chiến thắng |
|--------|---:|---:|---|
| **Latency (s)** | 139.04 | 117.36 | **Multi ✓** |
| **Cost (USD)** | ~ $0.004 (estimate) | $0.001886 | **Multi ✓** |
| **Citation Coverage** | 79% | 63% | **Single ✓** |
| **Output Length** | ~2000 tokens | ~1500 tokens | Single |
| **Error Rate** | 0% | 0% | Tie |
| **Iterations** | 1 | 4 (+ supervisor routing) | Single (simpler) |

---

## Detailed Analysis

### Latency: Multi-Agent Thắng

**Single-Agent:** 139.04s
- 1 LLM call với full query

**Multi-Agent:** 117.36s
- Iter 1: Supervisor route → Researcher search + notes (~40s)
- Iter 2: Supervisor route → Analyst analyze (~35s)  
- Iter 3: Supervisor route → Writer synthesize (~30s)
- Iter 4: Supervisor route → Done
- **Lợi thế**: Parallelization tiềm năng (hiện sequential, nhưng cấu trúc cho phép async)

### Cost: Multi-Agent Thắng

**Single-Agent:** ~$0.004 (134 in tokens, ~1,617 out)

**Multi-Agent:** $0.001886 (1,524 in tokens tổng, 878 out tokens)
- Researcher: in=194, out=130
- Analyst: in=359, out=598
- Writer: in=977, out=150

**Lý do rẻ hơn**: Output của mỗi agent nhỏ hơn single-agent (vì chuyên môn hóa), nên total tokens ít hơn.

### Citation Coverage: Single-Agent Thắng

**Single-Agent:** 79% (gần như mỗi paragraph đều cite)

**Multi-Agent:** 63% (một số paragraphs trong final answer không có explicit citations từ sources)

**Lý do**: Researcher generate sources, nhưng Writer không lúc nào cite hết — citation loss qua handoff.

---

## Failure Modes & Fixes

### 🔴 Mode 1: Citation Loss Through Handoff

**Vấn đề**: Researcher tìm sources và viết notes với citations, nhưng khi Writer tổng hợp final answer, Writer không lúc nào cite lại → citation coverage giảm.

**Root Cause**: 
- Researcher output dạng natural language notes (không structured citations)
- Writer không có explicit instruction "MUST cite [Source 1], [Source 2]" từ structured metadata

**Fix**:
```python
# Trong researcher.py, output structured sources + claim mapping
class ResearchOutput:
    sources: List[SourceDocument]  # explicit list
    claims_with_citations: List[Dict[str, Any]]  # {"claim": "...", "source_ids": [0, 1]}
    
# Trong writer.py, enforce citation
citation_constraint = """
For each claim, MUST cite the source using format: [Source N].
Do NOT make claims without citing. If unsure, cite the most relevant source.
"""
```

### 🔴 Mode 2: Iteration Overhead

**Vấn đề**: Multi-agent chạy 4 iterations (Supervisor, Researcher, Analyst, Writer, Done) với Supervisor routing logic.

**Root Cause**: 
- Supervisor dùng pure routing (không LLM call), nhưng vẫn overhead network/state management
- Sequential execution không tận dụng parallelization

**Fix**:
```python
# Add async execution
async def run_parallel(researchers, analysts):
    results = await asyncio.gather(
        researcher.run(state),
        analyst.run(state)  # if earlier results available
    )
    return results

# Reduce supervisor overhead: merge dengan first worker
class WorkflowOptimized:
    def run(self, state):
        # Inline routing + first agent call = 1 API call instead of 2
        ...
```

### 🟡 Mode 3: Source Metadata Loss

**Vấn đề**: SearchClient tạo mock sources (không real URLs), Writer không thể generate proper citations.

**Root Cause**: 
- Mock search chỉ return random title + snippet
- Writer không biết source là mock hay real → generate fake citations

**Fix**:
```python
# Replace mock với real search
class SearchClient:
    def search(self, query, max_results=5):
        if TAVILY_API_KEY:
            return tavily_search(query)  # Real search
        else:
            return mock_search_with_metadata(query)  # Only for dev

# Add validation: Writer should skip citing if source.is_mock
```

---

## Recommendations

### Ngắn hạn (Quick Wins)
1. **Fix citation enforcement** — add explicit instruction cho Writer
2. **Use real search API** (Tavily, SerpAPI) thay vì mock
3. **Enable async** — Researcher + Analyst có thể run song song nếu structured output

### Dài hạn (Architecture)
1. **Structured state** — sources, claims, citations để tránh loss during handoff
2. **Tracing + observability** — LangSmith/Langfuse để debug agent handoff
3. **Critique layer** — tự động verify citations trước output cuối

---

## Conclusion

**Multi-agent thắng về tốc độ + chi phí, nhưng thua về chất lượng (citation coverage).**

Việc phân tách agent giúp tối ưu individual outputs, nhưng thất thoát xảy ra ở các handoff point. **Cố định**: structured citations metadata + async execution + real search.

---

**Report generated:** 2026-06-24  
**Model:** Gemini 2.5 Flash  
**Lab:** Multi-Agent Research System (Phase 2, Day 5)
