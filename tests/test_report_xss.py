"""Tests for XSS prevention in the report generator."""

import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestReportXSSPrevention:
    """Verify that LLM-generated content is HTML-escaped in reports."""

    def _generate_report_html(self, companions_data):
        """Generate report HTML and return it as a string."""
        from scripts.generate_report import generate_html_report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name
        try:
            generate_html_report(companions_data, output_path)
            with open(output_path) as f:
                return f.read()  # noqa: the local 'html' in tests doesn't conflict
        finally:
            os.unlink(output_path)

    # Note: local variable 'html' in test methods is fine since we don't
    # import the html module here — the escaping happens in generate_report.py

    def _make_companion_data(self, **overrides):
        base = {
            'id': 'test',
            'messages': [],
            'opinions': [],
            'curiosities': [],
            'episodes': [],
            'reflections': [],
            'facts_count': 0,
        }
        base.update(overrides)
        return base

    def test_opinion_topic_escaped(self):
        data = self._make_companion_data(opinions=[
            {'topic': '<script>alert("xss")</script>', 'opinion_text': 'normal', 'confidence': 0.5}
        ])
        html = self._generate_report_html([data])
        assert '<script>alert("xss")</script>' not in html
        assert '&lt;script&gt;' in html

    def test_opinion_text_escaped(self):
        data = self._make_companion_data(opinions=[
            {'topic': 'safe', 'opinion_text': '<img onerror=alert(1) src=x>', 'confidence': 0.5}
        ])
        html = self._generate_report_html([data])
        # The raw <img> tag should be escaped — angle brackets become entities
        assert '<img onerror' not in html
        assert '&lt;img' in html

    def test_curiosity_topic_escaped(self):
        data = self._make_companion_data(curiosities=[
            {'topic': '"><script>alert(1)</script>', 'urgency': 0.5, 'resolved': False}
        ])
        html = self._generate_report_html([data])
        assert '<script>alert(1)</script>' not in html

    def test_episode_summary_escaped(self):
        data = self._make_companion_data(episodes=[
            {'summary': '<b onmouseover=alert(1)>hover</b>', 'learning': 'safe', 'created_at': '2026-01-01'}
        ])
        html = self._generate_report_html([data])
        # The raw <b> tag should be escaped — angle brackets become entities
        assert '<b onmouseover' not in html
        assert '&lt;b' in html

    def test_reflection_content_escaped(self):
        data = self._make_companion_data(reflections=[
            {'reflection_type': 'daily', 'content': '<iframe src="evil.com"></iframe>', 'created_at': '2026-01-01'}
        ])
        html = self._generate_report_html([data])
        assert '<iframe' not in html
