"""Tests for the code-first query classifier.

Verifies that development activity messages are correctly classified
into the right intent types for context budget allocation.
"""

import pytest
from engram.working.query_classifier import (
    classify_query,
    get_profile,
    ContextProfile,
    PROFILES,
)


# ---------------------------------------------------------------------------
# Profile structure tests
# ---------------------------------------------------------------------------


class TestProfiles:
    """Verify profile structure and budget constraints."""

    def test_all_profiles_have_valid_shares(self):
        """All profile shares should sum to <= 1.0."""
        for name, profile in PROFILES.items():
            total = (
                profile.identity_share
                + profile.relationship_share
                + profile.grounding_share
                + profile.recent_conversation_share
                + profile.episodic_share
                + profile.procedural_share
            )
            assert total <= 1.0 + 1e-9, f"Profile '{name}' shares sum to {total}"

    def test_all_profiles_have_positive_reserve(self):
        """All profiles should have some reserve (breathing room)."""
        for name, profile in PROFILES.items():
            assert profile.reserve_share > 0, f"Profile '{name}' has no reserve"

    def test_expected_profile_names(self):
        """Verify the code-first profile set matches buildplan."""
        expected = {
            "implementing",
            "debugging",
            "refactoring",
            "code_review",
            "code_navigation",
            "architecture",
            "general",
        }
        assert set(PROFILES.keys()) == expected

    def test_debugging_maximizes_episodic(self):
        """Debugging should have the highest episodic share."""
        debug_ep = PROFILES["debugging"].episodic_share
        for name, profile in PROFILES.items():
            if name != "debugging":
                assert debug_ep >= profile.episodic_share, (
                    f"debugging episodic ({debug_ep}) should be >= "
                    f"{name} ({profile.episodic_share})"
                )

    def test_refactoring_maximizes_procedural(self):
        """Refactoring should have the highest procedural share."""
        refactor_proc = PROFILES["refactoring"].procedural_share
        for name, profile in PROFILES.items():
            if name != "refactoring":
                assert refactor_proc >= profile.procedural_share, (
                    f"refactoring procedural ({refactor_proc}) should be >= "
                    f"{name} ({profile.procedural_share})"
                )

    def test_code_review_maximizes_grounding(self):
        """Code review should have the highest grounding (standards) share."""
        review_ground = PROFILES["code_review"].grounding_share
        for name, profile in PROFILES.items():
            if name != "code_review":
                assert review_ground >= profile.grounding_share, (
                    f"code_review grounding ({review_ground}) should be >= "
                    f"{name} ({profile.grounding_share})"
                )


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassifyQuery:
    """Test intent classification for development activities."""

    # --- Debugging ---

    def test_error_message(self):
        assert classify_query("fix the TypeError in auth.py") == "debugging"

    def test_stack_trace(self):
        assert (
            classify_query("I'm getting a traceback when running tests") == "debugging"
        )

    def test_broken_code(self):
        assert (
            classify_query("The build is broken after the last commit") == "debugging"
        )

    def test_not_working(self):
        assert classify_query("The login endpoint doesn't work anymore") == "debugging"

    def test_import_error(self):
        assert classify_query("ModuleNotFoundError when importing utils") == "debugging"

    # --- Implementing ---

    def test_add_feature(self):
        assert (
            classify_query("Add a new endpoint for user registration") == "implementing"
        )

    def test_create_class(self):
        assert (
            classify_query("Create a new class for handling payments") == "implementing"
        )

    def test_implement_function(self):
        assert (
            classify_query("Implement a retry function with exponential backoff")
            == "implementing"
        )

    def test_write_test(self):
        assert (
            classify_query("Write a test for the authentication module")
            == "implementing"
        )

    def test_build_feature(self):
        assert (
            classify_query("Build a caching layer for the API responses")
            == "implementing"
        )

    # --- Refactoring ---

    def test_refactor(self):
        assert (
            classify_query("Refactor the database connection handling") == "refactoring"
        )

    def test_clean_up(self):
        assert classify_query("Clean up the authentication middleware") == "refactoring"

    def test_extract_method(self):
        assert (
            classify_query("Extract the validation logic into a separate method")
            == "refactoring"
        )

    def test_simplify(self):
        assert (
            classify_query("Simplify the error handling in the API layer")
            == "refactoring"
        )

    def test_rename(self):
        assert (
            classify_query("Rename the getUserData function to fetchUser")
            == "refactoring"
        )

    # --- Code Review ---

    def test_review_code(self):
        assert (
            classify_query("Review this implementation for any issues") == "code_review"
        )

    def test_check_code(self):
        assert classify_query("Check my code for best practices") == "code_review"

    def test_any_issues(self):
        assert (
            classify_query("Are there any issues with this approach?") == "code_review"
        )

    def test_how_to_improve(self):
        assert classify_query("How can I improve this function?") == "code_review"

    # --- Code Navigation ---

    def test_where_is(self):
        assert (
            classify_query("Where is the authentication handler defined?")
            == "code_navigation"
        )

    def test_how_does_work(self):
        assert (
            classify_query("How does the caching system work in this project?")
            == "code_navigation"
        )

    def test_explain(self):
        assert (
            classify_query("Explain how the middleware chain processes requests")
            == "code_navigation"
        )

    def test_what_calls(self):
        assert (
            classify_query("What calls the validateToken function?")
            == "code_navigation"
        )

    # --- Architecture ---

    def test_architecture(self):
        assert (
            classify_query("What architecture should we use for the message queue?")
            == "architecture"
        )

    def test_design_decision(self):
        assert (
            classify_query("Should we use PostgreSQL or MongoDB for this feature?")
            == "architecture"
        )

    def test_tradeoffs(self):
        assert (
            classify_query("What are the trade-offs between REST and GraphQL?")
            == "architecture"
        )

    def test_adr(self):
        assert (
            classify_query("Create an ADR for the database migration strategy")
            == "architecture"
        )

    def test_scalability(self):
        assert classify_query("How can we make this more scalable?") == "architecture"

    # --- General ---

    def test_empty_string(self):
        assert classify_query("") == "general"

    def test_short_message(self):
        assert classify_query("hi") == "general"

    def test_ambiguous(self):
        assert classify_query("what's for lunch?") == "general"

    # --- Edge cases ---

    def test_long_message_classified_by_content(self):
        """Long multi-sentence messages should be classified by content."""
        # Contains "schema" -> architecture is reasonable
        long_msg = (
            "We need to update the user service to support multi-tenancy. "
            "This involves modifying the database schema, updating the ORM models, "
            "and changing the API endpoints. We also need to add tenant isolation "
            "to all queries and update the authentication middleware. "
            "Finally, we should write integration tests for the new behavior."
        )
        result = classify_query(long_msg)
        assert result in ("implementing", "architecture")

    def test_pure_long_message_defaults_to_implementing(self):
        """Long messages without specific pattern triggers use implementing."""
        long_msg = (
            "I want to add a new page to the application. "
            "It should have a form with several input fields. "
            "The form should validate all inputs before submission. "
            "After submission it should show a confirmation message. "
            "We also need to handle loading states and display progress."
        )
        assert classify_query(long_msg) == "implementing"

    def test_refactor_beats_implement(self):
        """'Refactor' should take priority over 'add'."""
        assert (
            classify_query("Refactor the add function to reduce complexity")
            == "refactoring"
        )

    def test_debugging_highest_priority(self):
        """Debugging should win over other patterns."""
        assert (
            classify_query("Fix the error in the new endpoint implementation")
            == "debugging"
        )


# ---------------------------------------------------------------------------
# get_profile integration
# ---------------------------------------------------------------------------


class TestGetProfile:
    """Test the classify + profile lookup convenience function."""

    def test_returns_profile(self):
        profile = get_profile("fix the bug in auth.py")
        assert isinstance(profile, ContextProfile)

    def test_debugging_profile_has_high_episodic(self):
        profile = get_profile("fix the TypeError in auth.py")
        assert profile.episodic_share >= 0.35

    def test_implementing_profile_balanced(self):
        profile = get_profile("Implement a new user registration endpoint")
        assert profile.procedural_share >= 0.20
        assert profile.episodic_share >= 0.15

    def test_unknown_returns_general(self):
        profile = get_profile("what's up")
        general = PROFILES["general"]
        assert profile == general
