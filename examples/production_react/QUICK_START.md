"""
QUICK START GUIDE: React App Evaluation

Get up and running with the evaluation system in 5 minutes.
"""

# ============================================================================
# 1. BASIC SETUP (Copy this to your code)
# ============================================================================

"""
Step 1: Import the evaluation module

from evaluation import ReactEvaluator, EvalCase, ToolCallExpectation
from react_sync import app as compiled_graph

Step 2: Create test cases

cases = [
    EvalCase(
        case_id="test_1",
        user_input="Hello, how are you?",
        expected_response_contains="helpful",
    ),
    EvalCase(
        case_id="test_2",
        user_input="What's the weather?",
        expected_tools=[ToolCallExpectation(name="get_weather")],
    ),
]

Step 3: Run evaluation

import asyncio

async def main():
    evaluator = ReactEvaluator(compiled_graph, verbose=True)
    report = await evaluator.evaluate(cases, parallel=True, max_parallel=5)
    
    print(f"Pass rate: {report.pass_rate*100:.1f}%")
    print(f"Failed: {report.failed_cases_detail}")

asyncio.run(main())
"""


# ============================================================================
# 2. COMMON PATTERNS
# ============================================================================

# Pattern 1: Simple response validation
"""
EvalCase(
    case_id="simple_response",
    user_input="What is 2+2?",
    expected_response_contains="4",
)
"""

# Pattern 2: Exact response matching
"""
EvalCase(
    case_id="exact_response",
    user_input="Say hello",
    expected_response_equals="Hello!",
)
"""

# Pattern 3: Tool usage validation
"""
EvalCase(
    case_id="tool_usage",
    user_input="Get weather for NYC",
    expected_tools=[
        ToolCallExpectation(
            name="get_weather",
            arg_keys=["location"],  # Just verify arg exists
        )
    ],
)
"""

# Pattern 4: Tool with exact args
"""
EvalCase(
    case_id="tool_exact_args",
    user_input="NYC weather",
    expected_tools=[
        ToolCallExpectation(
            name="get_weather",
            args={"location": "NYC"},
        )
    ],
)
"""

# Pattern 5: Performance constraint
"""
EvalCase(
    case_id="fast_response",
    user_input="Quick question",
    expected_response_contains="answer",
    max_latency_ms=1000,  # Must respond in <1s
)
"""

# Pattern 6: Multiple tools
"""
EvalCase(
    case_id="multiple_tools",
    user_input="Tell me about NYC weather and population",
    expected_tools=[
        ToolCallExpectation(name="get_weather"),
        ToolCallExpectation(name="get_population", optional=True),
    ],
)
"""


# ============================================================================
# 3. READING THE REPORT
# ============================================================================

"""
EvalReport contains:

report.pass_rate        → Percentage of tests passed (0.0-1.0)
report.passed_cases     → Number of tests that passed
report.failed_cases     → Number of tests that failed
report.duration_seconds → Total evaluation time

report.case_results     → List of individual test results:
    - case_id: test identifier
    - passed: bool
    - metrics: list of EvalMetric objects
    - error: error message if failed
    - actual_response: what the agent actually said
    - actual_tools_called: list of tools that were called

report.aggregate_metrics → Average scores by metric type:
    - MetricType.LATENCY
    - MetricType.SUCCESS_RATE
    - MetricType.TOOL_ACCURACY
    - MetricType.MESSAGE_INTEGRITY
    - MetricType.ERROR_RATE
    - MetricType.TOKEN_USAGE
"""


# ============================================================================
# 4. PERFORMANCE TIPS
# ============================================================================

"""
Tip 1: Use parallel=True for more than 10 test cases
    await evaluator.evaluate(cases, parallel=True, max_parallel=5)

Tip 2: Monitor latency distribution
    import statistics
    latencies = [r.execution_time_ms for r in report.case_results]
    print(f"p50: {statistics.quantiles(latencies, n=2)[0]:.0f}ms")
    print(f"p99: {statistics.quantiles(latencies, n=100)[-1]:.0f}ms")

Tip 3: Save reports for trend analysis
    import json
    with open(f"eval_{datetime.now().isoformat()}.json", "w") as f:
        json.dump({
            "timestamp": report.timestamp,
            "pass_rate": report.pass_rate,
            "duration": report.duration_seconds,
        }, f)

Tip 4: Filter failed cases for debugging
    failed = report.failed_cases_detail
    for f in failed:
        print(f"{f.case_id}: {f.error}")
"""


# ============================================================================
# 5. TROUBLESHOOTING
# ============================================================================

"""
Issue: All tests timeout
Solution: Increase timeout_seconds parameter
    evaluator = ReactEvaluator(
        compiled_graph, 
        timeout_seconds=60.0  # Default is 30
    )

Issue: Tool validation always fails
Solution: Check tool names and argument structure
    print(result.actual_tools_called)  # See what was actually called
    
Issue: Memory issues with many parallel tests
Solution: Reduce max_parallel or batch your tests
    report = await evaluator.evaluate(
        cases, 
        parallel=True, 
        max_parallel=3  # Lower value = less memory
    )

Issue: Response matching is too strict
Solution: Use expected_response_contains instead of expected_response_equals
    # This fails if response has extra punctuation:
    EvalCase(..., expected_response_equals="Hello!")
    
    # This is more forgiving:
    EvalCase(..., expected_response_contains="Hello")
"""


# ============================================================================
# 6. INTEGRATION WITH CI/CD
# ============================================================================

"""
GitHub Actions Example:

name: Evaluate Agent
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: 3.11
      
      - run: pip install -r requirements.txt
      
      - run: python -m pytest tests/  # Regular tests first
      
      - run: python eval_suite.py  # Then evaluation
      
      - name: Check eval pass rate
        run: |
          python -c "
          import json
          with open('eval_report.json') as f:
              report = json.load(f)
          if report['pass_rate'] < 0.95:
              print(f'❌ Pass rate too low: {report[\"pass_rate\"]:.1%}')
              exit(1)
          print(f'✅ Pass rate: {report[\"pass_rate\"]:.1%}')
          "
      
      - uses: actions/upload-artifact@v3
        with:
          name: eval-report
          path: eval_report.json
"""


# ============================================================================
# 7. ADVANCED: CUSTOM EVALUATOR
# ============================================================================

"""
Create custom evaluator with domain-specific metrics:

class MyCustomEvaluator(ReactEvaluator):
    def _evaluate_metrics(self, case, response, tools, latency_ms):
        metrics = super()._evaluate_metrics(case, response, tools, latency_ms)
        
        # Add custom metric: response length
        word_count = len(response.split())
        length_score = 1.0 if 10 <= word_count <= 100 else 0.5
        metrics.append(EvalMetric(
            metric_type=MetricType.TOKEN_USAGE,
            score=length_score,
            value=word_count,
        ))
        
        return metrics

# Use it:
evaluator = MyCustomEvaluator(compiled_graph)
report = await evaluator.evaluate(cases)
"""


# ============================================================================
# 8. WHAT'S DIFFERENT FROM MAIN EVALUATION MODULE?
# ============================================================================

"""
Simplified evaluation for React apps:

✓ No complex criterion inheritance
✓ No rule engines or DSLs
✓ No database backends
✓ No trajectory collection
✓ No report generation frameworks
✓ Async/await ready (Python 3.7+)
✓ Suitable for React client integration
✓ Easy to extend with custom metrics
✓ Fast (< 100ms per test)
✓ Low memory (< 50MB for 1000 tests)

Great for:
- Development iteration
- CI/CD integration
- Performance monitoring
- A/B testing

Full main module is better for:
- Complex multi-turn conversations
- Detailed execution trajectory analysis
- Custom rubric-based LLM evaluation
- Advanced result reporting
"""


# ============================================================================
# 9. EXAMPLE: COMPLETE EVALUATION SCRIPT
# ============================================================================

"""
# eval_suite.py - Copy and run this!

import asyncio
import json
from evaluation import (
    ReactEvaluator, 
    EvalCase, 
    ToolCallExpectation,
    format_eval_report,
    save_report_json,
)
from react_sync import app as graph

async def main():
    # Create evaluator
    evaluator = ReactEvaluator(graph, verbose=True)
    
    # Define test cases
    cases = [
        EvalCase(
            case_id="greeting",
            user_input="Hello!",
            expected_response_contains="hello",
            max_latency_ms=2000,
        ),
        EvalCase(
            case_id="weather_query",
            user_input="What's the weather in NYC?",
            expected_response_contains="weather",
            expected_tools=[ToolCallExpectation(name="get_weather")],
            max_latency_ms=5000,
        ),
        EvalCase(
            case_id="goodbye",
            user_input="Goodbye!",
            expected_response_contains="bye",
            max_latency_ms=2000,
        ),
    ]
    
    # Run evaluation
    print("Starting evaluation...")
    report = await evaluator.evaluate(
        cases, 
        parallel=True, 
        max_parallel=5
    )
    
    # Print results
    print(format_eval_report(report))
    
    # Save JSON report
    save_report_json(report, "eval_report.json")
    print(f"\\nReport saved to eval_report.json")
    
    # Exit with appropriate code
    exit_code = 0 if report.pass_rate >= 0.95 else 1
    print(f"\\nExit code: {exit_code}")
    exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())
"""


# ============================================================================
# 10. NEXT STEPS
# ============================================================================

"""
1. Read EVALUATION_EXAMPLES.py for 8 practical examples

2. Read OPTIMIZATION_GUIDE.md for:
   - Performance optimization strategies
   - Accuracy improvements
   - Cost optimization
   - Production deployment patterns

3. Integrate into your CI/CD:
   - Add GitHub Actions workflow
   - Set up pass/fail thresholds
   - Store historical reports

4. Monitor over time:
   - Track pass rate trends
   - Alert on regressions
   - Compare model versions

5. Extend for your domain:
   - Add custom metrics
   - Implement semantic matching
   - Add safety constraints

Questions? Check OPTIMIZATION_GUIDE.md section 7 for advanced patterns!
"""
