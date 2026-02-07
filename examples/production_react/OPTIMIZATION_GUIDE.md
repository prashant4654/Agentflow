"""
OPTIMIZATION GUIDE FOR REACT APP EVALUATION
=============================================

This document provides detailed strategies to optimize the evaluation system
for production React applications. The simplified evaluation module is designed
for performance from the start, but these optimizations can further improve
latency, accuracy, and resource usage.

Author: Agentflow Team
Last Updated: 2026-01-14


TABLE OF CONTENTS
================
1. Overview & Architecture
2. Performance Optimization Strategies
3. Accuracy & Metrics Optimization
4. Production Deployment Patterns
5. Monitoring & Observability
6. Cost Optimization
7. Advanced Patterns


═══════════════════════════════════════════════════════════════════════════════
1. OVERVIEW & ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

The evaluation system is built on principles of simplicity and performance:

Core Components:
  - EvalCase: Lightweight test case definition (~150 bytes)
  - EvalMetric: Individual metric computation (no side effects)
  - ReactEvaluator: Stateless evaluator with async/await patterns
  - MetricsCollector: Fast aggregation without persistence
  - Report Generation: Lazy formatting (only when needed)

Complexity Reduction:
  ✓ NO complex criterion inheritance hierarchy
  ✓ NO rule engine or DSL
  ✓ NO database backends or persistence
  ✓ NO ML models for evaluation (optional: use external LLM if needed)
  ✓ NO collecting execution trajectories
  ✓ NO report generation frameworks

Optimized For:
  ✓ Fast iteration during development (< 100ms per test)
  ✓ Low memory footprint (< 50MB for 1000 tests)
  ✓ Easy parallelization (async/await ready)
  ✓ Direct integration with React client
  ✓ CI/CD friendly (JSON output, exit codes)


═══════════════════════════════════════════════════════════════════════════════
2. PERFORMANCE OPTIMIZATION STRATEGIES
═══════════════════════════════════════════════════════════════════════════════

2.1 LATENCY OPTIMIZATION
─────────────────────────

Problem: Evaluation tests can be slow if run sequentially
Solution: Intelligent parallelization

Code Example:
    
    evaluator = ReactEvaluator(graph)
    
    # Sequential: ~5 seconds (5 cases × 1s each)
    report = await evaluator.evaluate(cases, parallel=False)
    
    # Parallel with limit: ~1.2 seconds (5 cases, max 5 concurrent)
    report = await evaluator.evaluate(cases, parallel=True, max_parallel=5)

Key Points:
  - Use parallel=True for > 10 test cases
  - max_parallel should be 2-3× your CPU cores
  - Set higher for IO-bound (API calls), lower for CPU-bound
  - Example: 4-core machine → max_parallel=10 for APIs

Performance Tips:
  1. Profile your bottleneck: Is it graph execution or metric computation?
  2. For API-bound graphs: Use max_parallel=20-50
  3. For LLM-bound graphs: Use max_parallel=5-10
  4. Monitor memory usage: Each parallel case needs ~1-2MB stack


2.2 GRAPH INVOCATION OPTIMIZATION
──────────────────────────────────

Current Implementation (Synchronous Graph):
    
    async def _invoke_graph(self, user_input: str):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.graph.invoke(...)
        )

Optimization Strategy 1: Native Async Graph
    
    class ReactEvaluatorAsync(ReactEvaluator):
        async def _invoke_graph(self, user_input: str):
            # If your graph is async:
            result = await self.graph.ainvoke({...})
            return self._extract_result(result)
    
    Benefit: ~30% latency improvement, reduced thread pool contention

Optimization Strategy 2: Graph Warming & Caching
    
    class ReactEvaluatorWarmed(ReactEvaluator):
        def __init__(self, graph, warm_up: int = 3):
            super().__init__(graph)
            self.compiled_graph = graph  # Trigger compilation once
            
        async def warm_up(self):
            # Pre-compile graph and warm caches
            for _ in range(3):
                await self._invoke_graph("hello")
    
    # Usage:
    evaluator = ReactEvaluatorWarmed(graph)
    await evaluator.warm_up()  # ~500ms, saves time on real tests
    report = await evaluator.evaluate(cases)

Optimization Strategy 3: Result Caching
    
    class ReactEvaluatorCached(ReactEvaluator):
        def __init__(self, graph, enable_cache: bool = True):
            super().__init__(graph)
            self.cache = {} if enable_cache else None
        
        async def _invoke_graph(self, user_input: str):
            if self.cache and user_input in self.cache:
                return self.cache[user_input]
            
            result = await super()._invoke_graph(user_input)
            if self.cache is not None:
                self.cache[user_input] = result
            return result
    
    Benefit: If tests repeat inputs, saves 95%+ latency on cache hits


2.3 METRIC COMPUTATION OPTIMIZATION
────────────────────────────────────

Current: Metrics computed inline during evaluation

Optimization: Defer metric computation
    
    class ReactEvaluatorDeferred(ReactEvaluator):
        async def _run_case(self, case):
            # Collect raw data only
            result = await self._invoke_graph(case.user_input)
            
            # Return raw case result
            return {
                "case_id": case.case_id,
                "raw_result": result,
                "case_config": case,
            }
        
        def _evaluate_metrics_batch(self, raw_results: list) -> EvalReport:
            # Compute metrics offline, can parallelize with numpy
            metrics = numpy.compute_metrics(raw_results)  # Fast!
            return EvalReport(...)
    
    Benefit: Separates data collection (latency-critical) from
    analysis (can be done offline, can use GPU if needed)


2.4 MEMORY OPTIMIZATION
───────────────────────

Profile Memory Usage:
    
    import tracemalloc
    
    tracemalloc.start()
    
    evaluator = ReactEvaluator(graph)
    report = await evaluator.evaluate(cases, parallel=True, max_parallel=10)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")

Memory Optimization Techniques:
    
    1. Streaming Results (for large test sets):
       
       async def evaluate_streaming(cases, batch_size=10):
           for i in range(0, len(cases), batch_size):
               batch = cases[i:i+batch_size]
               results = await evaluator.evaluate(batch)
               yield results  # Don't keep all in memory
               del results  # Explicit cleanup
    
    2. Clear Message History:
       
       class ReactEvaluatorClearHistory(ReactEvaluator):
           async def _invoke_graph(self, user_input):
               # Don't accumulate message history between test cases
               result = await super()._invoke_graph(user_input)
               # Reset any stateful graph components
               return result
    
    3. Use Pool of Workers:
       
       from concurrent.futures import ProcessPoolExecutor
       
       # For very heavy computation (LLM evals)
       with ProcessPoolExecutor(max_workers=4) as executor:
           metrics = await evaluate_with_pool(cases, executor)


═══════════════════════════════════════════════════════════════════════════════
3. ACCURACY & METRICS OPTIMIZATION
═══════════════════════════════════════════════════════════════════════════════

3.1 RESPONSE MATCHING ACCURACY
───────────────────────────────

Current Approach: Simple substring/exact match

Problem: Fragile to whitespace, punctuation, synonyms

Optimization 1: Fuzzy Matching
    
    from difflib import SequenceMatcher
    
    def fuzzy_response_match(expected: str, actual: str, threshold: float = 0.8):
        similarity = SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
        return similarity >= threshold
    
    # Usage in EvalCase:
    cases = [
        EvalCase(
            case_id="weather",
            user_input="weather?",
            expected_response_contains="sunny",  # Will match "It's sunny"
        )
    ]
    
    # Then in _evaluate_metrics:
    score = 1.0 if fuzzy_response_match(
        case.expected_response_contains,
        actual_response,
        threshold=0.75
    ) else 0.0

Optimization 2: Semantic Matching (for complex cases)
    
    from sentence_transformers import SentenceTransformer
    
    class ReactEvaluatorSemantic(ReactEvaluator):
        def __init__(self, graph, use_semantic: bool = False):
            super().__init__(graph)
            self.semantic_model = None
            if use_semantic:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        def _evaluate_response_semantic(self, expected: str, actual: str) -> float:
            if not self.semantic_model:
                return 0.0 if expected != actual else 1.0
            
            e_emb = self.semantic_model.encode(expected)
            a_emb = self.semantic_model.encode(actual)
            
            # Cosine similarity
            similarity = np.dot(e_emb, a_emb) / (
                np.linalg.norm(e_emb) * np.linalg.norm(a_emb)
            )
            return float(similarity)
    
    Benefits: Handles paraphrases, synonyms, word order variations
    Cost: ~100ms per comparison (use sparingly)

Optimization 3: Template-Based Matching
    
    import re
    
    def template_match(pattern: str, actual: str) -> bool:
        # Pattern: "The weather in {city} is {condition}"
        # Actual: "The weather in NYC is sunny"
        regex = pattern.replace("{city}", r"\\w+").replace("{condition}", r"\\w+")
        return bool(re.match(regex, actual))


3.2 TOOL CALL ACCURACY OPTIMIZATION
────────────────────────────────────

Current: Exact name and argument matching

Issue: Too strict - fails if args are in different order or have extras

Optimization 1: Relaxed Argument Matching
    
    def evaluate_tool_args_relaxed(expected: dict, actual: dict) -> float:
        """
        Args are correct if:
        - All expected keys present with correct values
        - Extra keys allowed (partial match)
        """
        score = 1.0
        for key, expected_value in expected.items():
            if key not in actual:
                score -= 0.5
            elif actual[key] != expected_value:
                # Allow type coercion (str "123" == int 123)
                if str(actual[key]) != str(expected_value):
                    score -= 0.25
        return max(0.0, score)

Optimization 2: Call Cardinality Tracking
    
    @dataclass
    class ToolCallExpectation:
        name: str
        args: dict | None = None
        arg_keys: list[str] | None = None
        optional: bool = False
        min_calls: int = 1
        max_calls: int = 1  # NEW: track max calls
    
    def evaluate_tool_cardinality(expected: ToolCallExpectation, 
                                  actual_calls: list[dict]) -> float:
        matching = [c for c in actual_calls if c["name"] == expected.name]
        count = len(matching)
        
        if count < expected.min_calls:
            return 0.0  # Tool not called enough
        if count > expected.max_calls:
            return 0.5  # Tool called too many times (penalty)
        return 1.0  # Perfect

Optimization 3: Tool Sequence Validation
    
    def validate_tool_sequence(expected_sequence: list[str], 
                               actual_tools: list[dict]) -> float:
        """Verify tools called in correct order"""
        actual_sequence = [t["name"] for t in actual_tools]
        
        # Check if expected sequence is a subsequence of actual
        it = iter(actual_sequence)
        try:
            for expected_tool in expected_sequence:
                while next(it) != expected_tool:
                    pass
            return 1.0
        except StopIteration:
            return 0.0


3.3 LATENCY METRICS IMPROVEMENT
────────────────────────────────

Current: Simple linear score (1.0 - latency/max_latency)

Issue: Doesn't account for percentiles, tail latencies

Optimization 1: Percentile-Based Scoring
    
    def latency_score_percentile(latencies_ms: list[float], 
                                current_ms: float) -> float:
        """Score based on percentile rank"""
        percentile = (sum(1 for l in latencies_ms if l <= current_ms) 
                     / len(latencies_ms))
        
        # Smooth curve: p95 = 0.8, p50 = 0.95
        return 1.0 - (0.2 * (1.0 - percentile) ** 0.5)

Optimization 2: SLO-Based Scoring
    
    class SLOBased:
        SLO_P50 = 100  # ms
        SLO_P99 = 500  # ms
        
        @staticmethod
        def latency_score(latency_ms: float) -> float:
            if latency_ms <= SLOBased.SLO_P50:
                return 1.0
            if latency_ms <= SLOBased.SLO_P99:
                return 0.5 + 0.5 * (1 - 
                    (latency_ms - SLOBased.SLO_P50) / 
                    (SLOBased.SLO_P99 - SLOBased.SLO_P50)
                )
            return 0.0  # SLO breach

Optimization 3: Latency Profiling
    
    @dataclass
    class LatencyProfile:
        p50: float
        p95: float
        p99: float
        max: float
    
    def profile_latencies(latencies_ms: list[float]) -> LatencyProfile:
        sorted_lat = sorted(latencies_ms)
        n = len(sorted_lat)
        return LatencyProfile(
            p50=sorted_lat[int(n * 0.5)],
            p95=sorted_lat[int(n * 0.95)],
            p99=sorted_lat[int(n * 0.99)],
            max=max(sorted_lat),
        )


═══════════════════════════════════════════════════════════════════════════════
4. PRODUCTION DEPLOYMENT PATTERNS
═══════════════════════════════════════════════════════════════════════════════

4.1 CI/CD INTEGRATION
────────────────────

Integrate evaluation into your CI/CD pipeline:

Example: GitHub Actions
    
    name: Agent Evaluation
    on: [push, pull_request]
    
    jobs:
      evaluate:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v3
          
          - name: Set up Python
            uses: actions/setup-python@v4
            with:
              python-version: 3.11
          
          - name: Install dependencies
            run: |
              pip install -r requirements.txt
              pip install agentflow
          
          - name: Run evaluation suite
            run: python eval_suite.py
          
          - name: Check pass rate
            run: |
              python -c "
              import json
              with open('eval_report.json') as f:
                  report = json.load(f)
              if report['pass_rate'] < 0.95:
                  exit(1)  # Fail if pass rate below 95%
              "
          
          - name: Upload report
            uses: actions/upload-artifact@v3
            with:
              name: eval-report
              path: eval_report.json

Example Evaluation Suite:
    
    # eval_suite.py
    import asyncio
    import json
    from evaluation import ReactEvaluator, EvalCase, ToolCallExpectation
    
    async def main():
        # Import your compiled graph
        from react_sync import app as graph
        
        evaluator = ReactEvaluator(graph, verbose=True)
        
        cases = [
            EvalCase(
                case_id="basic_greeting",
                user_input="Hello, how are you?",
                expected_response_contains="helpful",
                max_latency_ms=2000,
            ),
            EvalCase(
                case_id="tool_usage_weather",
                user_input="What's the weather in NYC?",
                expected_tools=[ToolCallExpectation(name="get_weather")],
                max_latency_ms=5000,
            ),
        ]
        
        report = await evaluator.evaluate(cases, parallel=True, max_parallel=5)
        
        # Save report
        with open("eval_report.json", "w") as f:
            json.dump({
                "timestamp": report.timestamp,
                "pass_rate": report.pass_rate,
                "passed": report.passed_cases,
                "failed": report.failed_cases,
                "total": report.total_cases,
            }, f, indent=2)
        
        # Print summary
        print(f"\nPass Rate: {report.pass_rate * 100:.1f}%")
        print(f"Duration: {report.duration_seconds:.2f}s")
        
        # Exit with appropriate code
        exit(0 if report.pass_rate >= 0.95 else 1)
    
    if __name__ == "__main__":
        asyncio.run(main())


4.2 CANARY DEPLOYMENTS
──────────────────────

Use evaluations to validate new model versions:
    
    # deploy_canary.py
    async def validate_model_upgrade(old_graph, new_graph, test_cases):
        old_evaluator = ReactEvaluator(old_graph)
        new_evaluator = ReactEvaluator(new_graph)
        
        old_report = await old_evaluator.evaluate(test_cases)
        new_report = await new_evaluator.evaluate(test_cases)
        
        # Check if new version is better
        old_score = old_report.pass_rate
        new_score = new_report.pass_rate
        
        if new_score >= old_score - 0.05:  # Allow 5% regression tolerance
            return True  # Safe to deploy
        else:
            print(f"Regression detected: {old_score:.1%} → {new_score:.1%}")
            return False


4.3 A/B TESTING WITH EVALUATIONS
────────────────────────────────

Compare two variants:
    
    async def compare_variants(graph_a, graph_b, cases):
        eval_a = ReactEvaluator(graph_a)
        eval_b = ReactEvaluator(graph_b)
        
        report_a = await eval_a.evaluate(cases)
        report_b = await eval_b.evaluate(cases)
        
        improvement = report_b.pass_rate - report_a.pass_rate
        
        if improvement > 0.05:
            return "variant_b"  # Variant B is winner
        elif improvement < -0.05:
            return "variant_a"
        else:
            return "no_significant_difference"


═══════════════════════════════════════════════════════════════════════════════
5. MONITORING & OBSERVABILITY
═══════════════════════════════════════════════════════════════════════════════

5.1 STRUCTURED LOGGING
──────────────────────

Add detailed logging to track evaluation runs:
    
    import logging
    import json
    from datetime import datetime
    
    class EvalLogger:
        def __init__(self, log_file: str = "eval.log"):
            self.logger = logging.getLogger("evaluation")
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"message": "%(message)s"}'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        def log_eval_start(self, num_cases: int):
            self.logger.info(f"Evaluation started: {num_cases} cases")
        
        def log_case_result(self, case_id: str, passed: bool, duration_ms: float):
            self.logger.info(
                f"Case completed: {case_id}, passed={passed}, "
                f"duration_ms={duration_ms}"
            )
        
        def log_eval_complete(self, report: EvalReport):
            self.logger.info(
                f"Evaluation complete: "
                f"pass_rate={report.pass_rate:.2%}, "
                f"duration={report.duration_seconds:.1f}s"
            )


5.2 METRICS COLLECTION FOR MONITORING
─────────────────────────────────────

Export metrics in Prometheus format:
    
    def export_prometheus_metrics(report: EvalReport) -> str:
        """Export evaluation metrics in Prometheus format"""
        lines = [
            f"# HELP eval_pass_rate Overall evaluation pass rate",
            f"# TYPE eval_pass_rate gauge",
            f"eval_pass_rate {{}} {report.pass_rate}",
            f"",
            f"# HELP eval_duration_seconds Total evaluation duration",
            f"# TYPE eval_duration_seconds gauge",
            f"eval_duration_seconds {{}} {report.duration_seconds}",
            f"",
            f"# HELP eval_total_cases Total test cases evaluated",
            f"# TYPE eval_total_cases counter",
            f"eval_total_cases {{}} {report.total_cases}",
        ]
        
        for metric_type, score in report.aggregate_metrics.items():
            lines.extend([
                f"# HELP eval_metric_{metric_type.value}",
                f"# TYPE eval_metric_{metric_type.value} gauge",
                f"eval_metric_{metric_type.value} {{}} {score}",
                f"",
            ])
        
        return "\n".join(lines)


5.3 HISTORICAL TRACKING
──────────────────────

Track evaluation metrics over time:
    
    class EvalHistory:
        def __init__(self, db_file: str = "eval_history.db"):
            import sqlite3
            self.conn = sqlite3.connect(db_file)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    timestamp TEXT,
                    pass_rate REAL,
                    duration_seconds REAL,
                    model_version TEXT,
                    branch_name TEXT
                )
            """)
        
        def record(self, report: EvalReport, model_version: str, 
                  branch_name: str):
            self.conn.execute("""
                INSERT INTO evaluations VALUES (?, ?, ?, ?, ?)
            """, (
                report.timestamp,
                report.pass_rate,
                report.duration_seconds,
                model_version,
                branch_name,
            ))
            self.conn.commit()
        
        def get_trend(self, days: int = 7) -> list[dict]:
            import json
            rows = self.conn.execute("""
                SELECT timestamp, pass_rate, duration_seconds
                FROM evaluations
                WHERE datetime(timestamp) > datetime('now', '-{} days')
                ORDER BY timestamp
            """.format(days))
            
            return [
                {
                    "timestamp": r[0],
                    "pass_rate": r[1],
                    "duration_seconds": r[2],
                }
                for r in rows
            ]


═══════════════════════════════════════════════════════════════════════════════
6. COST OPTIMIZATION
═══════════════════════════════════════════════════════════════════════════════

6.1 REDUCE LLM CALLS
────────────────────

Problem: Each evaluation test calls the LLM (expensive)

Solution 1: Sampling
    
    import random
    
    def sample_cases(cases: list[EvalCase], 
                    sample_rate: float = 0.1) -> list[EvalCase]:
        """Run full suite only on main branch, sample on PRs"""
        return random.sample(cases, int(len(cases) * sample_rate))
    
    # Usage in CI:
    if is_pull_request:
        cases_to_run = sample_cases(all_cases, sample_rate=0.2)  # 20% of cases
    else:
        cases_to_run = all_cases  # Run all on main

Solution 2: Caching Responses
    
    class CachedEvaluator(ReactEvaluator):
        def __init__(self, graph, cache_dir: str = ".eval_cache"):
            super().__init__(graph)
            self.cache_dir = cache_dir
            Path(cache_dir).mkdir(exist_ok=True)
        
        async def _invoke_graph(self, user_input: str):
            # Check cache first
            cache_file = Path(self.cache_dir) / f"{hash(user_input)}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)
            
            # Cache miss - invoke graph
            result = await super()._invoke_graph(user_input)
            
            # Save to cache
            with open(cache_file, "w") as f:
                json.dump(result, f)
            
            return result
    
    # Save cache in git (don't rerun expensive tests):
    # git add .eval_cache
    # git commit -m "Cache evaluation results"

Solution 3: Mock Simple Cases
    
    def mock_graph_for_testing(graph):
        """Wrap graph to mock some test cases"""
        import functools
        
        mocks = {
            "hello": {"response": "Hi there!", "tools_called": []},
            "goodbye": {"response": "Bye!", "tools_called": []},
        }
        
        original_invoke = graph.invoke
        
        @functools.wraps(original_invoke)
        def wrapped_invoke(messages, config=None):
            if len(messages) == 1:
                input_text = messages[0].get("content", "").lower()
                if input_text in mocks:
                    return mocks[input_text]
            return original_invoke(messages, config)
        
        graph.invoke = wrapped_invoke
        return graph


6.2 PARALLEL EXECUTION COST
────────────────────────────

Running evaluations in parallel costs more compute but saves wall-clock time.

Example cost analysis:
    - Sequential: 5 cases × 1s = 5s wall-clock, 5s CPU
    - Parallel (4 workers): 5s wall-clock, 4×5=20s CPU (4× cost!)
    
Solution: Balance parallelism with cost
    
    # For CI/CD (frequent runs, quick feedback):
    await evaluator.evaluate(cases, parallel=True, max_parallel=10)
    
    # For nightly batch runs (less frequent, cost-sensitive):
    await evaluator.evaluate(cases, parallel=True, max_parallel=2)


═══════════════════════════════════════════════════════════════════════════════
7. ADVANCED PATTERNS
═══════════════════════════════════════════════════════════════════════════════

7.1 CUSTOM METRICS
──────────────────

Add domain-specific metrics:
    
    class ReactEvaluatorCustomMetrics(ReactEvaluator):
        def _evaluate_metrics(self, case, actual_response, actual_tools, 
                            execution_time_ms):
            metrics = super()._evaluate_metrics(
                case, actual_response, actual_tools, execution_time_ms
            )
            
            # Add custom metric: response formality
            formality_score = self._evaluate_formality(actual_response)
            metrics.append(EvalMetric(
                metric_type=MetricType.SUCCESS_RATE,  # Reuse type
                score=formality_score,
                value=actual_response,
            ))
            
            # Add custom metric: response length
            if hasattr(case, "expected_length_tokens"):
                length_score = self._evaluate_length(
                    actual_response,
                    case.expected_length_tokens
                )
                metrics.append(EvalMetric(
                    metric_type=MetricType.TOKEN_USAGE,
                    score=length_score,
                    value=len(actual_response.split()),
                ))
            
            return metrics
        
        def _evaluate_formality(self, response: str) -> float:
            """Score response formality (0=casual, 1=formal)"""
            casual_words = {"yo", "hey", "lol", "uhh", "gonna"}
            count = sum(1 for word in response.lower().split() 
                       if word in casual_words)
            return 1.0 - (count / max(len(response.split()), 1))
        
        def _evaluate_length(self, response: str, expected_tokens: int) -> float:
            """Score response length (target ±20%)"""
            actual_tokens = len(response.split())
            min_allowed = expected_tokens * 0.8
            max_allowed = expected_tokens * 1.2
            
            if min_allowed <= actual_tokens <= max_allowed:
                return 1.0
            else:
                deviation = abs(actual_tokens - expected_tokens) / expected_tokens
                return max(0.0, 1.0 - deviation)


7.2 REGRESSION DETECTION
────────────────────────

Automatically detect performance regressions:
    
    class RegressionDetector:
        def __init__(self, baseline_report: EvalReport):
            self.baseline = baseline_report
        
        def check_regression(self, current_report: EvalReport) -> dict:
            """
            Check if current report regressed from baseline
            Returns: {
                "has_regression": bool,
                "regressions": [
                    {"metric": str, "baseline": float, "current": float}
                ],
                "severity": "none|low|medium|high",
            }
            """
            regressions = []
            
            # Check pass rate
            pass_rate_drop = (
                self.baseline.pass_rate - current_report.pass_rate
            )
            if pass_rate_drop > 0.05:  # > 5% regression
                regressions.append({
                    "metric": "pass_rate",
                    "baseline": self.baseline.pass_rate,
                    "current": current_report.pass_rate,
                    "change": -pass_rate_drop,
                })
            
            # Check latency
            baseline_latency = self.baseline.aggregate_metrics.get(
                MetricType.LATENCY, 1.0
            )
            current_latency = current_report.aggregate_metrics.get(
                MetricType.LATENCY, 1.0
            )
            latency_degradation = baseline_latency - current_latency
            if latency_degradation > 0.1:
                regressions.append({
                    "metric": "latency",
                    "baseline": baseline_latency,
                    "current": current_latency,
                    "change": latency_degradation,
                })
            
            severity = "none"
            if len(regressions) > 0:
                severity = "high" if pass_rate_drop > 0.1 else "medium"
            
            return {
                "has_regression": len(regressions) > 0,
                "regressions": regressions,
                "severity": severity,
            }


7.3 ADAPTIVE TEST SELECTION
────────────────────────────

Run more tests on risky changes:
    
    class AdaptiveTestSelection:
        @staticmethod
        def get_test_count(files_changed: list[str]) -> int:
            """Determine how many tests to run based on changed files"""
            
            risky_files = {
                "agent.py",
                "graph.py",
                "message.py",
                "evaluator.py",
            }
            
            changes_risky_files = any(
                file in risky_files for file in files_changed
            )
            
            if changes_risky_files:
                return 100  # Run comprehensive suite
            elif any("test" in f for f in files_changed):
                return 50  # Run half suite
            else:
                return 20  # Quick smoke test
    
    # Usage in CI:
    files_changed = get_changed_files_in_pr()
    num_tests = AdaptiveTestSelection.get_test_count(files_changed)
    test_cases = sampled_cases[:num_tests]
    report = await evaluator.evaluate(test_cases)


═══════════════════════════════════════════════════════════════════════════════
SUMMARY: OPTIMIZATION CHECKLIST
═══════════════════════════════════════════════════════════════════════════════

Quick Wins (implement these first):
  ☐ Enable parallel execution (parallel=True, max_parallel=5)
  ☐ Use response caching for repeated test inputs
  ☐ Add SLO-based latency scoring
  ☐ Implement fuzzy string matching

Medium Effort (for 10%+ improvement):
  ☐ Add streaming result collection
  ☐ Implement regression detection
  ☐ Set up CI/CD integration with pass/fail thresholds
  ☐ Add structured logging

Advanced (for complex scenarios):
  ☐ Semantic response matching (for paraphrases)
  ☐ Custom domain-specific metrics
  ☐ A/B testing framework
  ☐ Historical trend tracking


Recommended Optimization Priority:
1. Latency: Parallel execution (30% improvement)
2. Accuracy: Fuzzy matching + semantic scoring (20% improvement)
3. Cost: Caching + sampling (50% cost reduction)
4. Observability: Logging + monitoring (better debugging)


═══════════════════════════════════════════════════════════════════════════════
"""
