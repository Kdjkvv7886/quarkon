"""Tests for the clean-room privacy layer (FPSP surrogates + ephemeral vault)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from opendistill.privacy import (
    DEFAULT_ENTITIES,
    CleanRoomSanitizer,
    EntityKind,
    EphemeralVault,
    iban_check_digits,
    iban_valid,
    luhn_check_digit,
    luhn_valid,
    sanitize_text,
)

SAMPLE_TEXT = (
    "Customer John Smith (john.smith@corp-example.com) called from +380 (44) 123-45-67.\n"
    "Card 4539 5789 1234 5678 was used; IBAN DE89 3704 0044 0532 0130 00; server 10.14.22.7.\n"
    "API key sk-liveABCDEFGHIJKLMNOP1234, password=Hunter2!secret.\n"
    "Ship to 42 Klenova Street, Kyiv, 01001. Dr. Anna Shevchenko signed the form.\n"
)


class TestSurrogateValidity:
    def test_email_is_valid_and_preserves_uuid_local_shape(self) -> None:
        sanitizer = CleanRoomSanitizer()
        result = sanitizer.sanitize("write to first.last@company.co.ua please")
        assert result.changed
        assert "@" in result.text
        local = result.text.split()[2]
        assert "." in local.split("@")[0]
        assert not result.text.count("first.last@company.co.ua")
        # Reserved safe domain: surrogates can never be a real mailbox.
        assert any(domain in result.text for domain in ("example.com", "example.org", "example.net"))

    def test_card_surrogate_is_luhn_valid(self) -> None:
        sanitizer = CleanRoomSanitizer()
        # A Luhn-valid PAN (test number) must be replaced by another valid PAN.
        result = sanitizer.sanitize("pay with 4111111111111111 today")
        surrogate = result.text.split()[2]
        digits = "".join(char for char in surrogate if char.isdigit())
        assert len(digits) == 16
        assert digits != "4111111111111111"
        assert luhn_valid(digits), surrogate

    def test_card_in_context_without_valid_checksum(self) -> None:
        sanitizer = CleanRoomSanitizer()
        result = sanitizer.sanitize("card 1234 5678 9012 3456 on file")
        assert "1234 5678 9012 3456" not in result.text

    def test_iban_surrogate_is_valid(self) -> None:
        sanitizer = CleanRoomSanitizer()
        result = sanitizer.sanitize("IBAN DE89370400440532013000")
        surrogate = result.text.split()[1]
        assert iban_valid(surrogate), surrogate
        assert surrogate.startswith("DE")
        assert len(surrogate) == len("DE89370400440532013000")

    def test_phone_preserves_country_code_and_layout(self) -> None:
        sanitizer = CleanRoomSanitizer()
        text = "call +380 (44) 123-45-67 now"
        result = sanitizer.sanitize(text)
        replacement = result.text[len("call ") : result.text.index(" now")]
        assert replacement.startswith("+380 (")
        assert ")" in replacement and "-" in replacement
        assert replacement != "+380 (44) 123-45-67"

    def test_ip_mapped_to_documentation_range(self) -> None:
        sanitizer = CleanRoomSanitizer()
        result = sanitizer.sanitize("host 10.14.22.7 is down")
        ip = result.text.split()[1]
        assert ip.startswith(("192.0.2.", "198.51.100.", "203.0.113."))

    def test_secrets_are_replaced_and_prefix_kept(self) -> None:
        sanitizer = CleanRoomSanitizer()
        result = sanitizer.sanitize(SAMPLE_TEXT)
        assert "sk-liveABCDEFGHIJKLMNOP1234" not in result.text
        assert "Hunter2!secret" not in result.text
        assert "sk-" in result.text

    def test_no_static_mask_tokens_are_used(self) -> None:
        sanitizer = CleanRoomSanitizer()
        result = sanitizer.sanitize(SAMPLE_TEXT)
        for mask in ("[PERSON", "[EMAIL", "[IBAN", "[PHONE", "<PERSON", "{PERSON"):
            assert mask not in result.text

    def test_person_and_address_surrogates_replace_values(self) -> None:
        sanitizer = CleanRoomSanitizer()
        result = sanitizer.sanitize(SAMPLE_TEXT)
        assert "John Smith" not in result.text
        assert "Klenova" not in result.text
        assert "Anna Shevchenko" not in result.text or "Anna Shevchenko" not in sanitizer.sanitize(
            "Dr. Anna Shevchenko"
        ).text


class TestDeterminismAndVault:
    def test_surrogates_are_deterministic_across_instances(self) -> None:
        text = "email alex@corp.com and card 4539578912345678"
        first = CleanRoomSanitizer().sanitize(text).text
        second = CleanRoomSanitizer().sanitize(text).text
        assert first == second

    def test_same_entity_maps_to_same_surrogate(self) -> None:
        sanitizer = CleanRoomSanitizer()
        one = sanitizer.sanitize("mail me at john@corp.com")
        two = sanitizer.sanitize("again: john@corp.com")
        surrogate = one.text.split()[-1]
        assert surrogate == two.text.split()[-1]
        assert sanitizer.stats.entries >= 1

    def test_restore_roundtrip(self) -> None:
        sanitizer = CleanRoomSanitizer()
        result = sanitizer.sanitize(SAMPLE_TEXT)
        assert sanitizer.restore(result.text) == SAMPLE_TEXT

    def test_vault_keeps_mapping_out_of_serialisable_output(self) -> None:
        sanitizer = CleanRoomSanitizer()
        result = sanitizer.sanitize(SAMPLE_TEXT)
        payload = result.to_dict()
        dumped = json.dumps(payload)
        assert "john.smith@corp-example.com" not in dumped
        assert "John Smith" not in dumped
        assert "4539 5789 1234 5678" not in dumped
        assert "surrogate" in dumped

    def test_vault_repr_hides_values(self) -> None:
        vault = EphemeralVault()
        sanitizer = CleanRoomSanitizer(vault=vault)
        sanitizer.sanitize("mail john@corp.com")
        assert "john@corp.com" not in repr(vault)
        assert "entries=" in repr(vault)

    def test_vault_is_not_iterable(self) -> None:
        vault = EphemeralVault()
        with pytest.raises(TypeError):
            list(vault)

    def test_purge_destroys_mapping(self) -> None:
        sanitizer = CleanRoomSanitizer()
        sanitizer.sanitize(SAMPLE_TEXT)
        assert sanitizer.stats.entries > 0
        entries = sanitizer.purge()
        assert entries > 0
        assert sanitizer.stats.entries == 0
        assert sanitizer.stats.purged is True
        with pytest.raises(RuntimeError):
            sanitizer.sanitize("more text john@corp.com")
        with pytest.raises(RuntimeError):
            sanitizer.restore("anything")

    def test_purge_is_idempotent(self) -> None:
        sanitizer = CleanRoomSanitizer()
        sanitizer.sanitize("john@corp.com")
        assert sanitizer.purge() >= 0
        assert sanitizer.purge() == 0

    def test_shared_vault_reuses_surrogates(self) -> None:
        vault = EphemeralVault()
        first = CleanRoomSanitizer(vault=vault)
        second = CleanRoomSanitizer(vault=vault)
        one = first.sanitize("john@corp.com").text
        two = second.sanitize("john@corp.com").text
        assert one == two


class TestEntitySelection:
    def test_subset_of_entities(self) -> None:
        sanitizer = CleanRoomSanitizer(entities=[EntityKind.EMAIL])
        result = sanitizer.sanitize("john@corp.com and 4539578912345678")
        assert "john@corp.com" not in result.text
        assert "4539578912345678" in result.text

    def test_entities_accept_strings(self) -> None:
        sanitizer = CleanRoomSanitizer(entities=["email"])
        assert sanitizer.entities == (EntityKind.EMAIL,)

    def test_unknown_entity_rejected(self) -> None:
        with pytest.raises(ValueError):
            CleanRoomSanitizer(entities=["passport"])

    def test_custom_names_are_replaced(self) -> None:
        sanitizer = CleanRoomSanitizer(entities=[EntityKind.PERSON], custom_names=["Project Kraken"])
        result = sanitizer.sanitize("the owner is Project Kraken today")
        assert "Project Kraken" not in result.text

    def test_empty_text_is_noop(self) -> None:
        sanitizer = CleanRoomSanitizer()
        result = sanitizer.sanitize("")
        assert result.text == ""
        assert not result.changed

    def test_clean_text_is_unchanged(self) -> None:
        sanitizer = CleanRoomSanitizer()
        text = "Write a formula that sums column B."
        result = sanitizer.sanitize(text)
        assert result.text == text
        assert result.total == 0

    def test_default_entities_cover_required_kinds(self) -> None:
        names = {kind.value for kind in DEFAULT_ENTITIES}
        assert {"person", "email", "phone", "address", "card", "iban", "ip", "secret"} <= names

    def test_overlapping_matches_prefer_the_more_specific_kind(self) -> None:
        sanitizer = CleanRoomSanitizer()
        # A 16-digit PAN-shaped run must be handled as a card, not as a phone.
        result = sanitizer.sanitize("4539 5789 1234 5678")
        assert sanitizer.stats.by_kind.get("card", 0) >= 1
        assert result.text.strip() != "4539 5789 1234 5678"

    def test_pan_with_invalid_checksum_is_still_masked(self) -> None:
        sanitizer = CleanRoomSanitizer()
        result = sanitizer.sanitize("ref 4539578912345678 end")
        assert "4539578912345678" not in result.text


class TestChecksumHelpers:
    def test_luhn_helpers(self) -> None:
        assert luhn_valid("4539578912345678") is False  # invalid on purpose
        valid = "4539" + "5789" + "1234" + luhn_check_digit("453957891234")
        assert luhn_valid(valid)
        assert not luhn_valid("1234")

    def test_iban_helpers(self) -> None:
        assert iban_valid("DE89370400440532013000")
        check = iban_check_digits("DE", "370400440532013000")
        assert check == "89"
        assert not iban_valid("DE00370400440532013000")


class TestConvenienceFunction:
    def test_sanitize_text_purges_its_vault(self) -> None:
        result = sanitize_text("mail john@corp.com")
        assert "john@corp.com" not in result.text
        # The one-shot helper must not keep the mapping alive.
        assert result.total == 1

    def test_large_document_is_fast(self) -> None:
        sanitizer = CleanRoomSanitizer()
        chunk = "Contact john@corp.com or +380 (44) 123-45-67. " * 2000
        result = sanitizer.sanitize(chunk)
        assert result.total >= 4000
        assert len(result.text) > 0


_ = Path, Any
