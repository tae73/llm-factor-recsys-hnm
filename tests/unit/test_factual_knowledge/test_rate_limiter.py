"""Tests for TokenRateLimiter — sliding-window TPM rate limiter."""

import asyncio
import time

from src.knowledge.factual.extractor import TokenRateLimiter


class TestTokenRateLimiter:
    """Test sliding-window token rate limiter."""

    def test_acquire_under_budget_no_wait(self):
        """Acquire returns immediately when budget is sufficient."""

        async def _run():
            limiter = TokenRateLimiter(tpm_limit=100_000, window_seconds=60.0)
            limiter.record(1000)  # well under budget
            start = time.monotonic()
            await limiter.acquire()
            elapsed = time.monotonic() - start
            assert elapsed < 0.05, f"Expected <50ms, got {elapsed:.3f}s"

        asyncio.run(_run())

    def test_acquire_blocks_when_budget_exhausted(self):
        """Acquire blocks when window budget is exhausted."""

        async def _run():
            # Tiny budget + short window so test finishes fast
            limiter = TokenRateLimiter(tpm_limit=100, window_seconds=0.3)
            limiter.record(100)  # exhaust budget
            start = time.monotonic()
            await limiter.acquire()
            elapsed = time.monotonic() - start
            # Should wait ~0.3s (window) + 0.1s (padding) = ~0.4s
            assert elapsed >= 0.3, f"Expected >=0.3s wait, got {elapsed:.3f}s"

        asyncio.run(_run())

    def test_record_updates_running_average(self):
        """Running average reflects actual token usage after records."""
        limiter = TokenRateLimiter(tpm_limit=200_000)
        limiter.record(1000)
        limiter.record(2000)
        limiter.record(3000)
        # Average = (1000+2000+3000) / 3 = 2000
        assert limiter._avg_tokens == 2000

    def test_backpressure_pauses_acquire(self):
        """backpressure() causes acquire() to wait the specified duration."""

        async def _run():
            limiter = TokenRateLimiter(tpm_limit=1_000_000, window_seconds=60.0)
            limiter.backpressure(0.4)
            start = time.monotonic()
            await limiter.acquire()
            elapsed = time.monotonic() - start
            assert elapsed >= 0.35, f"Expected >=0.35s wait, got {elapsed:.3f}s"

        asyncio.run(_run())

    def test_sliding_window_expiry(self):
        """Budget recovers after window expires."""

        async def _run():
            limiter = TokenRateLimiter(tpm_limit=100, window_seconds=0.2)
            limiter.record(100)  # exhaust budget
            await asyncio.sleep(0.25)  # wait for window to expire
            start = time.monotonic()
            await limiter.acquire()
            elapsed = time.monotonic() - start
            # After expiry, should acquire quickly
            assert elapsed < 0.15, f"Expected <150ms after expiry, got {elapsed:.3f}s"

        asyncio.run(_run())

    def test_default_estimate_2500(self):
        """Conservative default of 2500 tokens when no history."""
        limiter = TokenRateLimiter(tpm_limit=200_000)
        assert limiter._avg_tokens == 2500

    def test_prune_removes_old_entries(self):
        """Old entries are pruned from the sliding window."""

        async def _run():
            limiter = TokenRateLimiter(tpm_limit=200_000, window_seconds=0.2)
            limiter.record(5000)
            assert len(limiter._log) == 1
            await asyncio.sleep(0.25)  # let entry expire
            now = time.monotonic()
            used = limiter._prune(now)
            assert used == 0, f"Expected 0 after prune, got {used}"
            assert len(limiter._log) == 0

        asyncio.run(_run())
