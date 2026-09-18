import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from flask import Flask


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from routes.episode_routes import bp
from routes.capture_routes import bp as capture_bp
from storage.episode_manager import EpisodeManager


def _make_episode(root: str, index: int) -> None:
    episode_id = f"episode_20260827_{index:06d}"
    episode_dir = os.path.join(root, episode_id)
    take_dir = os.path.join(episode_dir, "takes", "take_001")
    os.makedirs(take_dir)
    with open(os.path.join(episode_dir, "meta.json"), "w") as file:
        json.dump({
            "episode_id": episode_id,
            "created_at": f"2026-08-27T00:00:{index % 60:02d}",
            "task": {"instruction": f"task {index}"},
            "takes": ["take_001"],
        }, file)


class EpisodePaginationTests(unittest.TestCase):
    def test_lightweight_page_does_not_walk_directory_sizes(self):
        with tempfile.TemporaryDirectory() as root:
            for index in range(55):
                _make_episode(root, index)
            manager = EpisodeManager(root)
            with mock.patch.object(
                manager, "_dir_size_mb", side_effect=AssertionError("size walk called")
            ):
                page = manager.list_episodes(offset=24, limit=24, lightweight=True)

            self.assertEqual(manager.count_episodes(), 55)
            self.assertEqual(len(page), 24)
            self.assertNotIn("size_mb", page[0])
            self.assertNotIn("size_mb", page[0]["takes"][0])
            self.assertEqual(page[0]["episode_id"], "episode_20260827_000030")

    def test_paged_api_returns_envelope_and_legacy_api_keeps_array(self):
        with tempfile.TemporaryDirectory() as root:
            for index in range(5):
                _make_episode(root, index)
            app = Flask(__name__)
            app.config["CONTROLLER"] = types.SimpleNamespace(
                _episode=EpisodeManager(root)
            )
            app.register_blueprint(bp)
            client = app.test_client()

            response = client.get("/api/episodes?page=2&page_size=2")
            payload = response.get_json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload["page"], 2)
            self.assertEqual(payload["total"], 5)
            self.assertEqual(len(payload["items"]), 2)
            self.assertTrue(payload["has_more"])

            legacy = client.get("/api/episodes").get_json()
            self.assertIsInstance(legacy, list)
            self.assertEqual(len(legacy), 5)

    def test_replay_rejection_uses_conflict_status_and_reason(self):
        controller = types.SimpleNamespace(
            start_replay=lambda speed_scale=None: {
                "ok": False,
                "reason": "speed exceeds safety limit",
            }
        )
        app = Flask(__name__)
        app.config["CONTROLLER"] = controller
        app.register_blueprint(capture_bp)
        response = app.test_client().post(
            "/api/replay/start", json={"speed_scale": 0.7}
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["reason"], "speed exceeds safety limit")


if __name__ == "__main__":
    unittest.main()
