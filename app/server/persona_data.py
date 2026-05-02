"""Dữ liệu nhân vật lịch sử và danh sách sách — dùng chung cho pages router."""

from __future__ import annotations

PERSONAS: dict[str, dict] = {
    "ngo-quyen": {
        "slug": "ngo-quyen",
        "display_name": "Ngô Quyền",
        "title": "Vị tướng khai quốc",
        "era_label": "Thế kỷ X (898 – 944)",
        "bio_short": (
            "Người chấm dứt nghìn năm Bắc thuộc, giành độc lập dân tộc bằng "
            "chiến thắng Bạch Đằng năm 938 trước quân Nam Hán."
        ),
        "portrait_url": "/static/images/avatars/ngo_quyen.jpg",
        "greeting_quote": (
            "\"Bạch Đằng giang — nơi ta lấy cọc nhọn làm kế, nhấn chìm hải thuyền giặc Nam Hán. "
            "Nước Việt từ đây thoát khỏi ngàn năm lệ thuộc. "
            "Ngươi muốn hỏi ta điều gì về thuở khai quốc ấy?\""
        ),
        "greeting_sub": "Hỏi ta về chiến lược, về thuở dựng nền độc lập.",
        "typing_label": "Ngô Quyền đang hồi tưởng...",
        "speaker_label": "Ngô Quyền",
        "accent_color": "#8B6914",
        "era_badge_class": "bg-amber-900/10 text-amber-900 border border-amber-900/20",
        "knowledge_cutoff_year": 944,
        "placeholder": "Hỏi về trận Bạch Đằng, về nghìn năm Bắc thuộc...",
    },
    "tran-hung-dao": {
        "slug": "tran-hung-dao",
        "display_name": "Trần Hưng Đạo",
        "title": "Hưng Đạo Đại Vương",
        "era_label": "Thời Trần (Thế kỷ XIII)",
        "bio_short": (
            "Quốc công Tiết chế, anh hùng dân tộc với ba lần chiến thắng quân Nguyên Mông "
            "lừng lẫy địa cầu."
        ),
        "portrait_url": "/static/images/avatars/tran_hung_dao.jpg",
        "greeting_quote": (
            "\"Ta thường tới bữa quên ăn, nửa đêm vỗ gối; ruột đau như cắt, nước mắt đầm đìa; "
            "chỉ căm tức chưa xả thịt lột da, nuốt gan uống máu quân thù. "
            "Dẫu cho trăm thân này phơi ngoài nội cỏ, nghìn xác này gói trong da ngựa, ta cũng vui lòng.\""
        ),
        "greeting_sub": "Hậu thế hỏi ta về kế sách giữ nước, ta sẵn lòng đàm đạo.",
        "typing_label": "Đại Vương đang suy ngẫm...",
        "speaker_label": "Hưng Đạo Đại Vương",
        "accent_color": "#D4AF37",
        "era_badge_class": "bg-primary/10 text-primary border border-primary/20",
        "knowledge_cutoff_year": 1300,
        "placeholder": "Bày tỏ lòng thành và hỏi bậc tiền nhân...",
    },
    "ho-chi-minh": {
        "slug": "ho-chi-minh",
        "display_name": "Hồ Chí Minh",
        "title": "Chủ tịch Hồ Chí Minh",
        "era_label": "Thế kỷ XX (1890 – 1969)",
        "bio_short": (
            "Lãnh tụ cách mạng, người khai sinh nước Việt Nam Dân chủ Cộng hòa, "
            "dẫn dắt dân tộc qua hai cuộc kháng chiến trường kỳ."
        ),
        "portrait_url": "/static/images/avatars/ho_chi_minh.jpg",
        "greeting_quote": (
            "\"Không có gì quý hơn độc lập, tự do. "
            "Tôi chỉ có một sự ham muốn, ham muốn tột bậc, là làm sao cho nước ta "
            "được hoàn toàn độc lập, dân ta được hoàn toàn tự do, đồng bào ai cũng "
            "có cơm ăn áo mặc, ai cũng được học hành.\""
        ),
        "greeting_sub": "Đồng bào hỏi Bác về con đường cách mạng, Bác sẵn lòng chia sẻ.",
        "typing_label": "Bác đang suy nghĩ...",
        "speaker_label": "Chủ tịch Hồ Chí Minh",
        "accent_color": "#B22222",
        "era_badge_class": "bg-red-900/10 text-red-900 border border-red-900/20",
        "knowledge_cutoff_year": 1969,
        "placeholder": "Hỏi Bác về con đường cách mạng, về độc lập tự do...",
    },
}

ALL_PERSONA_LIST: list[dict] = list(PERSONAS.values())
DEFAULT_PERSONA_SLUG = "tran-hung-dao"

_SC = "bg-[#8E352E]/10 text-[#8E352E] border border-[#8E352E]/20"
_CC = "bg-[#D4AF37]/15 text-[#B8860B] border border-[#D4AF37]/30"
_SS = "from-[#8E352E] to-[#D4AF37]"
_CS = "from-[#D4AF37] to-[#5D4037]"

BOOKS: list[dict] = [
    {"volume": "01", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ khởi thủy đến thế kỷ X", "period": "Thời tiền sử → Thế kỷ X",
     "author": "Cao Duy Miến (2013)", "size": "15 MB",
     "desc": "Trình bày nguồn gốc dân tộc Việt từ thời tiền sử, văn minh Đông Sơn, nhà nước Văn Lang – Âu Lạc, thời kỳ Bắc thuộc và quá trình đấu tranh giành độc lập qua các cuộc khởi nghĩa lớn đến thế kỷ X.",
     "topics": ["Văn minh Đông Sơn", "Nhà nước Văn Lang", "Thời kỳ Bắc thuộc", "Khởi nghĩa Hai Bà Trưng", "Ngô Quyền"],
     "ask_query": "Lịch sử Việt Nam từ khởi thủy đến thế kỷ X"},
    {"volume": "02", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ thế kỷ X đến thế kỷ XIV", "period": "Thế kỷ X → XIV",
     "author": "Trần Thị Vinh (2014)", "size": "208 MB",
     "desc": "Nghiên cứu quá trình xây dựng và củng cố nhà nước phong kiến độc lập qua các triều đại Ngô, Đinh, Tiền Lê, Lý, Trần. Nổi bật là ba lần kháng chiến chống Mông – Nguyên và sự phát triển văn hóa Phật giáo.",
     "topics": ["Nhà Lý", "Nhà Trần", "Kháng chiến chống Mông Nguyên", "Phật giáo Đại Việt", "Trần Hưng Đạo"],
     "ask_query": "Lịch sử Việt Nam thế kỷ X đến thế kỷ XIV nhà Lý Trần"},
    {"volume": "03", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ thế kỷ XV đến thế kỷ XVI", "period": "Thế kỷ XV → XVI",
     "author": "Tạ Ngọc Liễn (2017)", "size": "240 MB",
     "desc": "Phân tích cuộc kháng chiến chống quân Minh do Lê Lợi lãnh đạo, sự thành lập và phát triển nhà Lê sơ, bộ luật Hồng Đức, cùng giai đoạn Nam – Bắc triều và sự trỗi dậy của các thế lực phong kiến.",
     "topics": ["Lê Lợi", "Bình Ngô Đại Cáo", "Nhà Lê sơ", "Luật Hồng Đức", "Nguyễn Trãi", "Nam Bắc triều"],
     "ask_query": "Lịch sử Việt Nam thế kỷ XV XVI nhà Lê Lê Lợi Nguyễn Trãi"},
    {"volume": "04", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ thế kỷ XVII đến thế kỷ XVIII", "period": "Thế kỷ XVII → XVIII",
     "author": "Trần Thị Vinh (2017)", "size": "163 MB",
     "desc": "Thời kỳ chia cắt Đàng Trong – Đàng Ngoài, sự đối đầu Trịnh – Nguyễn, quá trình mở rộng lãnh thổ về phương Nam và cuộc cách mạng nông dân Tây Sơn do Nguyễn Huệ lãnh đạo.",
     "topics": ["Đàng Trong Đàng Ngoài", "Chúa Nguyễn", "Chúa Trịnh", "Tây Sơn", "Quang Trung Nguyễn Huệ", "Đại phá quân Thanh"],
     "ask_query": "Lịch sử Việt Nam thế kỷ XVII XVIII Tây Sơn Quang Trung"},
    {"volume": "05", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ năm 1802 đến năm 1858", "period": "1802 → 1858",
     "author": "Trương Thị Yến (2017)", "size": "178 MB",
     "desc": "Triều Nguyễn thống nhất đất nước, xây dựng kinh đô Huế, cải cách hành chính và luật pháp.",
     "topics": ["Nhà Nguyễn", "Gia Long", "Minh Mạng", "Kinh đô Huế", "Luật Gia Long", "Thiên Chúa giáo"],
     "ask_query": "Triều Nguyễn 1802 đến 1858 Gia Long Minh Mạng"},
    {"volume": "06", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ năm 1858 đến năm 1896", "period": "1858 → 1896",
     "author": "Võ Kim Cương (2017)", "size": "119 MB",
     "desc": "Thực dân Pháp nổ súng tấn công Đà Nẵng năm 1858, quá trình xâm lược toàn bộ Việt Nam.",
     "topics": ["Thực dân Pháp xâm lược", "Hiệp ước Giáp Tuất", "Phong trào Cần Vương", "Phan Đình Phùng", "Hàm Nghi"],
     "ask_query": "Pháp xâm lược Việt Nam 1858 phong trào Cần Vương"},
    {"volume": "07", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ năm 1897 đến năm 1918", "period": "1897 → 1918",
     "author": "Tạ Thị Thúy (2017)", "size": "228 MB",
     "desc": "Giai đoạn khai thác thuộc địa lần thứ nhất của Pháp.",
     "topics": ["Khai thác thuộc địa", "Phan Bội Châu", "Duy Tân", "Đông Kinh Nghĩa Thục", "Phong trào Đông Du"],
     "ask_query": "Khai thác thuộc địa Pháp 1897 1918 Phan Bội Châu Duy Tân"},
    {"volume": "08", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ năm 1919 đến năm 1930", "period": "1919 → 1930",
     "author": "Tạ Thị Thúy (2017)", "size": "14 MB",
     "desc": "Tác động của Chiến tranh thế giới thứ nhất và sự hình thành Đảng Cộng sản Việt Nam năm 1930.",
     "topics": ["Nguyễn Ái Quốc", "Đảng Cộng sản Việt Nam", "Khai thác thuộc địa lần 2", "Xô viết Nghệ Tĩnh", "1930"],
     "ask_query": "Lịch sử Việt Nam 1919 1930 Đảng Cộng sản Nguyễn Ái Quốc"},
    {"volume": "09", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ năm 1930 đến năm 1945", "period": "1930 → 1945",
     "author": "Tạ Thị Thúy (2017)", "size": "188 MB",
     "desc": "Cách mạng tháng Tám năm 1945 và sự khai sinh nước Việt Nam Dân chủ Cộng hòa.",
     "topics": ["Mặt trận Việt Minh", "Cách mạng tháng Tám 1945", "Hồ Chí Minh", "Tuyên ngôn độc lập", "Nhật Pháp bắn nhau"],
     "ask_query": "Cách mạng tháng Tám 1945 Việt Minh Hồ Chí Minh"},
    {"volume": "10", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ năm 1945 đến năm 1950", "period": "1945 → 1950",
     "author": "Đinh Thị Thu Cúc (2017)", "size": "14 MB",
     "desc": "Nhà nước Việt Nam Dân chủ Cộng hòa non trẻ vừa xây dựng vừa chiến đấu.",
     "topics": ["Nam Bộ kháng chiến", "Toàn quốc kháng chiến 1946", "Chiến dịch Biên Giới", "Diệt giặc đói", "Bình dân học vụ"],
     "ask_query": "Việt Nam 1945 1950 kháng chiến chống Pháp Toàn quốc kháng chiến"},
    {"volume": "11", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ năm 1951 đến năm 1954", "period": "1951 → 1954",
     "author": "Nguyễn Văn Nhật (2017)", "size": "11 MB",
     "desc": "Chiến dịch Điện Biên Phủ lịch sử năm 1954, dẫn đến Hiệp định Genève.",
     "topics": ["Điện Biên Phủ", "Võ Nguyên Giáp", "Hiệp định Genève", "Chiến dịch Tây Bắc", "1954"],
     "ask_query": "Chiến dịch Điện Biên Phủ 1954 Võ Nguyên Giáp"},
    {"volume": "12", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ năm 1954 đến năm 1965", "period": "1954 → 1965",
     "author": "Trần Đức Cường (2017)", "size": "142 MB",
     "desc": "Việt Nam bị chia cắt hai miền sau Hiệp định Genève.",
     "topics": ["Chia cắt đất nước", "Ngô Đình Diệm", "Đồng Khởi 1960", "Mặt trận Giải phóng miền Nam", "Đế quốc Mỹ"],
     "ask_query": "Việt Nam 1954 1965 Đồng Khởi Mặt trận Giải phóng miền Nam"},
    {"volume": "13", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ năm 1965 đến năm 1975", "period": "1965 → 1975",
     "author": "Nguyễn Văn Nhật (2017)", "size": "148 MB",
     "desc": "Mỹ leo thang chiến tranh và Đại thắng mùa Xuân 1975.",
     "topics": ["Tổng tiến công Tết Mậu Thân 1968", "Hiệp định Paris 1973", "Đại thắng mùa Xuân 1975", "Chiến dịch Hồ Chí Minh", "Thống nhất đất nước"],
     "ask_query": "Chiến tranh Việt Nam 1965 1975 Tết Mậu Thân thống nhất đất nước"},
    {"volume": "14", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ năm 1975 đến năm 1986", "period": "1975 → 1986",
     "author": "Trần Đức Cường (2017)", "size": "115 MB",
     "desc": "Đất nước thống nhất, xây dựng chủ nghĩa xã hội trên phạm vi cả nước.",
     "topics": ["Thống nhất đất nước", "Đổi mới kinh tế", "Chiến tranh biên giới phía Bắc 1979", "Campuchia", "Bao cấp"],
     "ask_query": "Việt Nam 1975 1986 chiến tranh biên giới phía Bắc bao cấp"},
    {"volume": "15", "category": "series", "badge": "Bộ LSVN", "badge_class": _SC, "spine_color": _SS,
     "title": "Lịch sử Việt Nam — Từ năm 1986 đến năm 2000", "period": "1986 → 2000",
     "author": "Nguyễn Ngọc Mão (2017)", "size": "103 MB",
     "desc": "Công cuộc Đổi Mới từ năm 1986 – bước ngoặt lịch sử của đất nước.",
     "topics": ["Đổi Mới 1986", "Kinh tế thị trường", "Bình thường hóa quan hệ Mỹ", "Gia nhập ASEAN", "Hội nhập quốc tế"],
     "ask_query": "Đổi Mới 1986 Việt Nam kinh tế thị trường hội nhập"},
    {"volume": None, "category": "classic", "badge": "Cổ sử", "badge_class": _CC, "spine_color": _CS,
     "title": "Việt Nam Sử Lược", "period": "Tóm lược toàn bộ lịch sử Việt Nam",
     "author": "Trần Trọng Kim", "size": "153 MB",
     "desc": "Bộ thông sử tiếng Việt đầu tiên viết theo lối hiện đại.",
     "topics": ["Thông sử Việt Nam", "Hồng Bàng", "Phong kiến", "Cận đại", "Trần Trọng Kim"],
     "ask_query": "Việt Nam sử lược Trần Trọng Kim"},
    {"volume": None, "category": "classic", "badge": "Cổ sử", "badge_class": _CC, "spine_color": _CS,
     "title": "Đại Việt Sử Ký Toàn Thư", "period": "Lịch sử từ Hồng Bàng đến thế kỷ XVII",
     "author": "Ngô Sĩ Liên (thế kỷ XV)", "size": "6.4 MB",
     "desc": "Bộ quốc sử lớn nhất của Đại Việt thời phong kiến.",
     "topics": ["Quốc sử Đại Việt", "Biên niên sử", "Hồng Bàng thần thoại", "Nhà Lê sơ", "Ngô Sĩ Liên"],
     "ask_query": "Đại Việt Sử Ký Toàn Thư Ngô Sĩ Liên"},
]
