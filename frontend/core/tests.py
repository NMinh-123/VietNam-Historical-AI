import json
from unittest.mock import AsyncMock, Mock, patch

from django.test import TestCase
from django.urls import reverse


class PersonaChatApiTests(TestCase):
    def test_persona_chat_api_returns_answer_with_sources(self):
        engine = Mock()
        engine.ask_with_sources = AsyncMock(
            return_value={
                "answer": "Trần Hưng Đạo chỉ huy kháng chiến chống Nguyên Mông [nguon=1].",
                "sources": [
                    {
                        "index": 1,
                        "title": "Lịch sử Việt Nam tập 1",
                        "file_name": "lich-su-viet-nam-tap-1.pdf",
                        "file_path": "/tmp/lich-su-viet-nam-tap-1.pdf",
                        "page": 120,
                        "page_label": "121",
                        "parent_id": "parent_000123",
                        "score": 4.25,
                        "label": "lich-su-viet-nam-tap-1.pdf, trang 121",
                    }
                ],
                "verification": "Trả lời dựa trên 1 nguồn vector đã truy hồi.",
            }
        )

        with patch("core.views._get_query_engine", return_value=engine):
            response = self.client.post(
                reverse("persona_chat_api"),
                data=json.dumps({"question": "Trần Hưng Đạo là ai?"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["answer"],
            "Trần Hưng Đạo chỉ huy kháng chiến chống Nguyên Mông [nguon=1].",
        )
        self.assertEqual(payload["sources"][0]["page_label"], "121")
        self.assertEqual(
            payload["verification"],
            "Trả lời dựa trên 1 nguồn vector đã truy hồi.",
        )
        engine.ask_with_sources.assert_awaited_once_with("Trần Hưng Đạo là ai?")
