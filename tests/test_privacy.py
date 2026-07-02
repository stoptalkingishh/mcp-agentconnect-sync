from agentconnect.common.privacy import ClassificationHints, classify, redact
from agentconnect.common.schemas import PrivacyClass


def test_secret_key_classified_secret_sensitive():
    payload = "here is my key sk-ABCD1234EFGH5678IJKL and more"
    assert classify(payload) == PrivacyClass.secret_sensitive


def test_denylisted_file_is_secret_sensitive():
    hints = ClassificationHints(file_paths=(".env",))
    assert classify("PORT=8080", hints) == PrivacyClass.secret_sensitive


def test_private_repo_is_repo_sensitive():
    hints = ClassificationHints(from_private_repo=True)
    assert classify("def foo(): pass", hints) == PrivacyClass.repo_sensitive


def test_email_is_low_sensitive():
    assert classify("contact jane.doe@example.com") == PrivacyClass.low_sensitive


def test_plain_text_is_public():
    assert classify("What is the capital of France?") == PrivacyClass.public


def test_declared_class_overrides():
    hints = ClassificationHints(declared=PrivacyClass.restricted)
    assert classify("anything", hints) == PrivacyClass.restricted


def test_redaction_removes_secret_and_marks_not_cloud_safe():
    payload = "token sk-ABCD1234EFGH5678IJKL used in code"
    result, redacted = redact(payload, PrivacyClass.secret_sensitive)
    assert "sk-ABCD1234EFGH5678IJKL" not in redacted
    assert result.cloud_safe is False
    assert result.redactions


def test_redaction_low_sensitive_is_cloud_safe_after_scrub():
    payload = "email me at a@b.com and visit host.internal"
    result, redacted = redact(payload, PrivacyClass.low_sensitive)
    assert "a@b.com" not in redacted
    assert result.cloud_safe is True
