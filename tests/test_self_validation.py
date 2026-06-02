from unittest.mock import patch, MagicMock
from src.core.conversation.response_critic import ResponseCritic, QualityCritique


def test_critic_fetches_strategy_tips_when_user_email_provided():
    """ResponseCritic.critique() should fetch strategy tips when user_email is given."""
    critic = ResponseCritic()
    mock_tips = [
        MagicMock(content="User dislikes unsolicited advice"),
        MagicMock(content="User responds well to humor"),
    ]

    with patch('src.core.conversation.response_critic.get_strategy_tip_store') as mock_store_fn:
        mock_store = MagicMock()
        mock_store.retrieve_relevant_tips.return_value = mock_tips
        mock_store_fn.return_value = mock_store

        with patch(
            'src.core.conversation.response_critic.generate_sync',
            return_value='SCORE: 8\nGENERIC: no\nADDRESSES: yes\nREGISTER: yes\nREPETITIVE: no\nSUGGESTION: none'
        ):
            try:
                critic.critique(
                    user_email='t@t.com',
                    user_message='hello',
                    response='hey there',
                )
            except Exception:
                pass  # May fail on parsing, that's fine

    mock_store.retrieve_relevant_tips.assert_called()


def test_critic_works_without_strategy_tips():
    """ResponseCritic must not fail when strategy tips are unavailable."""
    critic = ResponseCritic()

    with patch('src.core.conversation.response_critic.get_strategy_tip_store',
               side_effect=Exception("DB unavailable")):
        with patch(
            'src.core.conversation.response_critic.generate_sync',
            return_value='SCORE: 7\nGENERIC: no\nADDRESSES: yes\nREGISTER: yes\nREPETITIVE: no\nSUGGESTION: none'
        ):
            try:
                result = critic.critique(
                    user_email='t@t.com',
                    user_message='hi',
                    response='hey',
                )
                # Should not raise regardless of tip fetch failure
            except Exception as e:
                assert False, f"critique() raised unexpectedly: {e}"


from src.core.conversation.pipeline import ConversationPipeline


def test_emotional_coherence_passes_when_state_empty():
    """Empty internal_state means no constraint — always passes."""
    pipeline = ConversationPipeline()
    assert pipeline._check_emotional_coherence(
        response="WOW I'M SO EXCITED!!!",
        internal_state="",
    ) is True


def test_emotional_coherence_passes_on_exception():
    """Must return True (pass) if the LLM call fails — never block pipeline."""
    from unittest.mock import patch
    pipeline = ConversationPipeline()
    with patch.object(pipeline, '_llm_coherence_check', side_effect=Exception("API down")):
        result = pipeline._check_emotional_coherence(
            response="test response",
            internal_state="exhausted",
        )
    assert result is True
