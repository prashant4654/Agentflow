"""
PRACTICAL EXAMPLES: Using the React App Evaluation System

This file demonstrates real-world usage patterns for the simplified
evaluation module with the React production agent.

Examples included:
1. Basic evaluation with simple test cases
2. Tool usage validation
3. Parallel evaluation for performance
4. Report generation and monitoring
5. Integration with CI/CD
6. Custom metrics and extensions
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

# From evaluation.py
from evaluation import (
    ReactEvaluator,
    EvalCase,
    EvalMetric,
    ToolCallExpectation,
    EvalReport,
    MetricType,
    format_eval_report,
    save_report_json,
)

# From react_sync.py
from react_sync import app as compiled_graph


# ============================================================================
# EXAMPLE 1: BASIC EVALUATION
# ============================================================================


async def example_basic_evaluation():
    """
    Simplest usage: Create test cases and run evaluation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Evaluation")
    print("=" * 70)

    evaluator = ReactEvaluator(compiled_graph, verbose=True)

    # Define simple test cases
    cases = [
        EvalCase(
            case_id="greeting",
            user_input="Hello, how are you?",
            expected_response_contains="helpful",
            description="Test basic greeting response",
        ),
        EvalCase(
            case_id="goodbye",
            user_input="Goodbye",
            expected_response_contains="bye",
            description="Test goodbye response",
        ),
    ]

    # Run evaluation
    report = await evaluator.evaluate(cases)

    # Display results
    print(format_eval_report(report))


# ============================================================================
# EXAMPLE 2: TOOL USAGE VALIDATION
# ============================================================================


async def example_tool_validation():
    """
    Test that agent calls the right tools with correct arguments
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Tool Usage Validation")
    print("=" * 70)

    evaluator = ReactEvaluator(compiled_graph, verbose=True)

    cases = [
        EvalCase(
            case_id="weather_tool_call",
            user_input="What's the weather in New York City?",
            expected_response_contains="weather",
            expected_tools=[
                ToolCallExpectation(
                    name="get_weather",
                    arg_keys=["location"],  # Just check location arg exists
                )
            ],
            max_latency_ms=5000,
            description="Should call get_weather tool",
        ),
        EvalCase(
            case_id="weather_with_exact_args",
            user_input="Get weather for London",
            expected_response_contains="London",
            expected_tools=[
                ToolCallExpectation(
                    name="get_weather",
                    args={"location": "London"},  # Exact match
                )
            ],
            description="Should call get_weather with location=London",
        ),
        EvalCase(
            case_id="multiple_tools",
            user_input="Tell me about NYC weather and population",
            expected_tools=[
                ToolCallExpectation(name="get_weather", min_calls=1),
                # Could add population tool if available
            ],
            description="May call multiple tools",
        ),
    ]

    report = await evaluator.evaluate(cases)
    print(format_eval_report(report))

    # Analyze tool usage
    for result in report.case_results:
        print(f"\nCase: {result.case_id}")
        print(f"  Tools called: {result.actual_tools_called}")
        print(
            f"  Response: {result.actual_response[:100] if result.actual_response else 'None'}..."
        )


# ============================================================================
# EXAMPLE 3: PERFORMANCE OPTIMIZATION WITH PARALLELIZATION
# ============================================================================


async def example_parallel_evaluation():
    """
    Run evaluation in parallel for faster feedback
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Parallel Evaluation (Performance)")
    print("=" * 70)

    evaluator = ReactEvaluator(compiled_graph, verbose=False)

    # Create many test cases
    cases = [
        EvalCase(
            case_id=f"test_{i}",
            user_input=f"Question {i}: What's the weather?",
            expected_response_contains="weather",
            max_latency_ms=3000,
        )
        for i in range(20)
    ]

    # Sequential evaluation (slow)
    print("\nSequential evaluation...")
    import time

    start = time.time()
    report_seq = await evaluator.evaluate(cases, parallel=False)
    seq_time = time.time() - start
    print(f"Duration: {seq_time:.2f}s")

    # Parallel evaluation (fast)
    print("\nParallel evaluation (max_parallel=5)...")
    start = time.time()
    report_par = await evaluator.evaluate(cases, parallel=True, max_parallel=5)
    par_time = time.time() - start
    print(f"Duration: {par_time:.2f}s")

    print(f"\nSpeedup: {seq_time / par_time:.1f}x faster with parallelization")
    print(f"Pass rate: {report_par.pass_rate * 100:.1f}%")


# ============================================================================
# EXAMPLE 4: REPORT GENERATION AND MONITORING
# ============================================================================


async def example_report_generation():
    """
    Generate different report formats for monitoring
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Report Generation")
    print("=" * 70)

    evaluator = ReactEvaluator(compiled_graph, verbose=False)

    cases = [
        EvalCase(
            case_id="test_1",
            user_input="Hello?",
            expected_response_contains="response",
        ),
        EvalCase(
            case_id="test_2",
            user_input="Weather?",
            expected_response_contains="weather",
        ),
        EvalCase(
            case_id="test_3",
            user_input="Tool test",
            expected_tools=[ToolCallExpectation(name="get_weather")],
        ),
    ]

    report = await evaluator.evaluate(cases, parallel=True, max_parallel=3)

    # 1. Console report
    print("\n--- CONSOLE REPORT ---")
    print(format_eval_report(report))

    # 2. JSON report (for monitoring systems)
    print("\n--- JSON REPORT ---")
    save_report_json(report, "eval_report.json")
    print("Saved to eval_report.json")

    # 3. Custom formatted report for CI/CD
    print("\n--- CI/CD SUMMARY ---")
    print(f"PASS_RATE: {report.pass_rate * 100:.1f}%")
    print(f"PASSED: {report.passed_cases}/{report.total_cases}")
    print(f"DURATION: {report.duration_seconds:.2f}s")

    # 4. Metrics by type
    print("\n--- METRICS BREAKDOWN ---")
    for metric_type, score in report.aggregate_metrics.items():
        print(f"{metric_type.value:20} {score * 100:6.1f}%")

    # 5. Failed cases summary
    if report.failed_cases_detail:
        print("\n--- FAILED CASES ---")
        for failed in report.failed_cases_detail:
            print(f"  • {failed.case_id}: {failed.error or 'Metric not met'}")


# ============================================================================
# EXAMPLE 5: REGRESSION DETECTION
# ============================================================================


async def example_regression_detection():
    """
    Compare baseline vs. current evaluation to detect regressions
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Regression Detection")
    print("=" * 70)

    evaluator = ReactEvaluator(compiled_graph, verbose=False)

    cases = [
        EvalCase(case_id="test_1", user_input="q1", expected_response_contains="a"),
        EvalCase(case_id="test_2", user_input="q2", expected_response_contains="b"),
        EvalCase(case_id="test_3", user_input="q3", expected_response_contains="c"),
    ]

    # Simulate baseline (previous good state)
    print("Running baseline evaluation...")
    baseline_report = await evaluator.evaluate(cases)
    baseline_pass_rate = baseline_report.pass_rate

    # Simulate current (after code change)
    print("Running current evaluation...")
    current_report = await evaluator.evaluate(cases)
    current_pass_rate = current_report.pass_rate

    # Compare
    print(f"\nBaseline pass rate: {baseline_pass_rate * 100:.1f}%")
    print(f"Current pass rate:  {current_pass_rate * 100:.1f}%")

    regression = baseline_pass_rate - current_pass_rate
    if regression > 0.05:
        print(f"\n⚠️  REGRESSION DETECTED: {regression * 100:.1f}% drop")
        print("Recommendation: Revert changes or fix issues")
    elif regression < -0.05:
        print(f"\n✅ IMPROVEMENT: {-regression * 100:.1f}% gain")
        print("Safe to merge!")
    else:
        print(f"\n✓  No significant regression (Δ = {regression * 100:.1f}%)")


# ============================================================================
# EXAMPLE 6: SAMPLING FOR COST REDUCTION
# ============================================================================


async def example_sampling():
    """
    Use sampling to reduce costs on PR checks while keeping full suite
    for main branch
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Intelligent Sampling (Cost Optimization)")
    print("=" * 70)

    import random

    # Create large test suite
    all_cases = [
        EvalCase(
            case_id=f"test_{i}",
            user_input=f"Test query {i}",
            expected_response_contains="response",
        )
        for i in range(100)
    ]

    evaluator = ReactEvaluator(compiled_graph, verbose=False)

    # Simulate branch detection (in CI: check $BRANCH_NAME env var)
    is_pull_request = True

    if is_pull_request:
        # Sample 20% for quick feedback on PRs
        sample_rate = 0.2
        test_cases = random.sample(all_cases, int(len(all_cases) * sample_rate))
        print(f"PR detected: Running sampled suite ({len(test_cases)} cases)")
    else:
        # Run full suite on main branch
        test_cases = all_cases
        print(f"Main branch: Running full suite ({len(test_cases)} cases)")

    report = await evaluator.evaluate(test_cases, parallel=True, max_parallel=10)

    print(f"Pass rate: {report.pass_rate * 100:.1f}%")
    print(f"Duration: {report.duration_seconds:.2f}s")

    # Estimate full suite performance
    if is_pull_request:
        estimated_full_time = report.duration_seconds / sample_rate
        print(f"Estimated full suite time: {estimated_full_time:.2f}s")
        print(f"Time saved: {estimated_full_time - report.duration_seconds:.2f}s")


# ============================================================================
# EXAMPLE 7: CUSTOM METRICS
# ============================================================================


class CustomReactEvaluator(ReactEvaluator):
    """
    Extended evaluator with custom domain-specific metrics
    """

    def _evaluate_metrics(self, case, actual_response, actual_tools, execution_time_ms):
        # Get standard metrics from parent
        metrics = super()._evaluate_metrics(case, actual_response, actual_tools, execution_time_ms)

        # Add custom metric: response politeness
        politeness_score = self._evaluate_politeness(actual_response)
        metrics.append(
            EvalMetric(
                metric_type=MetricType.SUCCESS_RATE,  # Reuse existing type
                score=politeness_score,
                value=actual_response,
            )
        )

        # Add custom metric: response conciseness
        conciseness_score = self._evaluate_conciseness(actual_response)
        metrics.append(
            EvalMetric(
                metric_type=MetricType.TOKEN_USAGE,
                score=conciseness_score,
                value=len(actual_response.split()),
            )
        )

        return metrics

    def _evaluate_politeness(self, response: str) -> float:
        """Score politeness: uses polite words like 'please', 'thank you'"""
        polite_words = {"please", "thank", "appreciate", "sorry", "excuse"}
        count = sum(1 for word in response.lower().split() if any(p in word for p in polite_words))
        return min(1.0, count * 0.2)  # Max 5 polite words = 1.0

    def _evaluate_conciseness(self, response: str) -> float:
        """Score conciseness: prefer 10-100 word responses"""
        word_count = len(response.split())
        if 10 <= word_count <= 100:
            return 1.0
        else:
            deviation = abs(word_count - 50) / 50
            return max(0.0, 1.0 - deviation)


async def example_custom_metrics():
    """
    Demonstrate custom metrics
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Custom Metrics")
    print("=" * 70)

    evaluator = CustomReactEvaluator(compiled_graph, verbose=True)

    cases = [
        EvalCase(
            case_id="polite_response",
            user_input="Can you help me please?",
            expected_response_contains="help",
            description="Should have polite response",
        ),
        EvalCase(
            case_id="concise_response",
            user_input="Tell me briefly",
            expected_response_contains="response",
            description="Should be concise",
        ),
    ]

    report = await evaluator.evaluate(cases)

    print("\nCustom metrics are included in the evaluation")
    for result in report.case_results:
        print(f"\n{result.case_id}:")
        for metric in result.metrics:
            print(f"  {metric.metric_type.value}: {metric.score:.2f}")


# ============================================================================
# EXAMPLE 8: CI/CD INTEGRATION TEMPLATE
# ============================================================================


async def example_ci_cd_integration():
    """
    Example of how to integrate evaluation into CI/CD pipeline
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: CI/CD Integration Template")
    print("=" * 70)

    print("""
# Sample CI/CD script (e.g., .github/workflows/eval.yml)

import asyncio
import sys
import json
from evaluation import ReactEvaluator, EvalCase

async def main():
    # Import graph
    from react_sync import app as graph
    
    evaluator = ReactEvaluator(graph, verbose=True)
    
    # Define critical test cases
    critical_cases = [
        EvalCase(
            case_id="basic_functionality",
            user_input="Test basic input",
            expected_response_contains="response",
        ),
        # ... more critical cases
    ]
    
    # Run evaluation
    report = await evaluator.evaluate(
        critical_cases, 
        parallel=True, 
        max_parallel=5
    )
    
    # Save report
    with open("eval_report.json", "w") as f:
        json.dump({
            "pass_rate": report.pass_rate,
            "passed": report.passed_cases,
            "total": report.total_cases,
        }, f)
    
    # Exit with appropriate code
    if report.pass_rate >= 0.95:
        print("✅ Evaluation passed!")
        sys.exit(0)
    else:
        print(f"❌ Evaluation failed (pass rate: {report.pass_rate*100:.1f}%)")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
    """)


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================


async def main():
    """Run all examples"""

    print("\n" + "=" * 80)
    print("REACT PRODUCTION APP EVALUATION SYSTEM - PRACTICAL EXAMPLES")
    print("=" * 80)

    try:
        await example_basic_evaluation()
        await example_tool_validation()
        await example_parallel_evaluation()
        await example_report_generation()
        await example_regression_detection()
        await example_sampling()
        await example_custom_metrics()
        await example_ci_cd_integration()

        print("\n" + "=" * 80)
        print("All examples completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
