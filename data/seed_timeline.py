#!/usr/bin/env python3
"""
Seed timeline database with Vietnamese dynasties and kings.
Run from the frontend directory:
    python3 ../data/seed_timeline.py
"""
import os, sys, django

# ── Setup Django ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../frontend")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "frontend.settings")
django.setup()

from core.models import Dynasty, King

DB = "timeline"

# Xoá dữ liệu cũ
King.objects.using(DB).all().delete()
Dynasty.objects.using(DB).all().delete()

# ── Data ──────────────────────────────────────────────────────────────────────
DYNASTIES = [
    {
        "slug": "hung-vuong",
        "name": "Thời Hùng Vương",
        "start_year": -2879,
        "end_year": -258,
        "color": "#C0392B",
        "description": "18 đời Hùng Vương dựng nước Văn Lang, nền tảng của văn minh Lạc Việt.",
        "kings": [
            {"name": "Kinh Dương Vương", "reign_start": -2879, "reign_end": -2794, "temple_name": "Lộc Tục"},
            {"name": "Lạc Long Quân", "reign_start": -2793, "reign_end": -2525, "temple_name": "Sùng Lãm"},
            {"name": "Hùng Vương thứ I", "reign_start": -2524, "reign_end": -2253},
            {"name": "Hùng Vương thứ II", "reign_start": -2252, "reign_end": -1913},
            {"name": "Hùng Vương thứ III", "reign_start": -1912, "reign_end": -1713},
            {"name": "Hùng Vương thứ IV", "reign_start": -1712, "reign_end": -1632},
            {"name": "Hùng Vương thứ V", "reign_start": -1631, "reign_end": -1432},
            {"name": "Hùng Vương thứ VI", "reign_start": -1431, "reign_end": -1332},
            {"name": "Hùng Vương thứ VII", "reign_start": -1331, "reign_end": -1132},
            {"name": "Hùng Vương thứ VIII", "reign_start": -1131, "reign_end": -1032},
            {"name": "Hùng Vương thứ IX", "reign_start": -1031, "reign_end": -932},
            {"name": "Hùng Vương thứ X", "reign_start": -931, "reign_end": -832},
            {"name": "Hùng Vương thứ XI", "reign_start": -831, "reign_end": -732},
            {"name": "Hùng Vương thứ XII", "reign_start": -731, "reign_end": -632},
            {"name": "Hùng Vương thứ XIII", "reign_start": -631, "reign_end": -532},
            {"name": "Hùng Vương thứ XIV", "reign_start": -531, "reign_end": -432},
            {"name": "Hùng Vương thứ XV", "reign_start": -431, "reign_end": -332},
            {"name": "Hùng Vương thứ XVI", "reign_start": -331, "reign_end": -258, "description": "Vua cuối cùng thời Hùng Vương, nhường ngôi cho Thục Phán."},
        ],
    },
    {
        "slug": "nha-thuc",
        "name": "Nhà Thục (Âu Lạc)",
        "start_year": -257,
        "end_year": -207,
        "color": "#8E44AD",
        "description": "Thục Phán (An Dương Vương) thống nhất Văn Lang và Âu Việt, lập nước Âu Lạc, xây thành Cổ Loa.",
        "kings": [
            {"name": "An Dương Vương", "reign_start": -257, "reign_end": -207, "temple_name": "Thục Phán",
             "description": "Xây thành Cổ Loa, chế nỏ thần. Mất nước vào tay Triệu Đà năm 207 TCN."},
        ],
    },
    {
        "slug": "bac-thuoc",
        "name": "Thời Bắc thuộc",
        "start_year": -207,
        "end_year": 938,
        "color": "#7F8C8D",
        "description": "Hơn 1000 năm Bắc thuộc dưới các triều đại Trung Hoa. Nhiều cuộc khởi nghĩa anh hùng.",
        "kings": [
            {"name": "Hai Bà Trưng", "reign_start": 40, "reign_end": 43, "description": "Khởi nghĩa đánh đuổi Đông Hán, xưng vương."},
            {"name": "Bà Triệu", "reign_start": 248, "reign_end": 248, "description": "Khởi nghĩa chống nhà Ngô, 'Muốn cưỡi cơn gió mạnh, đạp luồng sóng dữ'."},
            {"name": "Lý Bí (Lý Nam Đế)", "reign_start": 544, "reign_end": 548, "description": "Lập nước Vạn Xuân, xưng hoàng đế đầu tiên."},
            {"name": "Triệu Quang Phục", "reign_start": 548, "reign_end": 571, "description": "Giữ vững nước Vạn Xuân sau Lý Nam Đế."},
            {"name": "Khúc Thừa Dụ", "reign_start": 905, "reign_end": 907, "description": "Tự chủ đầu tiên sau thời Bắc thuộc."},
            {"name": "Khúc Hạo", "reign_start": 907, "reign_end": 917},
            {"name": "Dương Đình Nghệ", "reign_start": 931, "reign_end": 937, "description": "Đánh tan quân Nam Hán, giành quyền tự chủ."},
        ],
    },
    {
        "slug": "nha-ngo",
        "name": "Nhà Ngô",
        "start_year": 939,
        "end_year": 965,
        "color": "#2980B9",
        "description": "Ngô Quyền chiến thắng quân Nam Hán trên sông Bạch Đằng năm 938, mở ra kỷ nguyên độc lập.",
        "kings": [
            {"name": "Ngô Quyền", "reign_start": 939, "reign_end": 944, "temple_name": "Ngô Vương",
             "description": "Đại thắng Bạch Đằng 938. Đóng đô ở Cổ Loa. Khai sáng nền độc lập lâu dài."},
            {"name": "Dương Tam Kha", "reign_start": 944, "reign_end": 950, "description": "Chiếm quyền sau khi Ngô Quyền mất."},
            {"name": "Ngô Xương Ngập", "reign_start": 950, "reign_end": 954, "description": "Con trai Ngô Quyền, khôi phục ngôi nhà Ngô."},
            {"name": "Ngô Xương Văn", "reign_start": 950, "reign_end": 965, "description": "Đồng trị với anh, thời kỳ suy yếu dẫn đến loạn 12 sứ quân."},
        ],
    },
    {
        "slug": "nha-dinh",
        
        "name": "Nhà Đinh",
        "start_year": 968,
        "end_year": 980,
        "color": "#D35400",
        "description": "Đinh Bộ Lĩnh dẹp loạn 12 sứ quân, thống nhất đất nước, đặt quốc hiệu Đại Cồ Việt.",
        "kings": [
            {"name": "Đinh Tiên Hoàng", "reign_start": 968, "reign_end": 979, "temple_name": "Đinh Bộ Lĩnh",
             "description": "Dẹp 12 sứ quân, lập Đại Cồ Việt, đặt kinh đô Hoa Lư, xưng Hoàng đế."},
            {"name": "Đinh Phế Đế", "reign_start": 979, "reign_end": 980, "temple_name": "Đinh Toàn",
             "description": "Lên ngôi khi còn nhỏ, mẹ là Dương Vân Nga trao quyền cho Lê Hoàn."},
        ],
    },
    {
        "slug": "nha-tien-le",
        "name": "Nhà Tiền Lê",
        "start_year": 980,
        "end_year": 1009,
        "color": "#27AE60",
        "description": "Lê Hoàn đánh tan quân Tống, giữ vững nền độc lập. Mở đầu cho kỷ nguyên phục hưng.",
        "kings": [
            {"name": "Lê Đại Hành", "reign_start": 980, "reign_end": 1005, "temple_name": "Lê Hoàn",
             "description": "Đại thắng quân Tống 981, đánh dẹp Chiêm Thành. Trị vì 25 năm thái bình thịnh vượng."},
            {"name": "Lê Trung Tông", "reign_start": 1005, "reign_end": 1005, "temple_name": "Lê Long Việt"},
            {"name": "Lê Ngoạ Triều", "reign_start": 1005, "reign_end": 1009, "temple_name": "Lê Long Đĩnh",
             "description": "Vua tàn bạo, thời kỳ hỗn loạn. Mất năm 1009, Lý Công Uẩn lên thay."},
        ],
    },
    {
        "slug": "nha-ly",
        "name": "Nhà Lý",
        "start_year": 1009,
        "end_year": 1225,
        "color": "#F39C12",
        "description": "Thời kỳ hưng thịnh dài 216 năm. Dời đô ra Thăng Long, phát triển Phật giáo và văn hóa.",
        "kings": [
            {"name": "Lý Thái Tổ", "reign_start": 1009, "reign_end": 1028, "temple_name": "Lý Công Uẩn",
             "description": "Dời đô từ Hoa Lư ra Đại La, đổi tên thành Thăng Long năm 1010."},
            {"name": "Lý Thái Tông", "reign_start": 1028, "reign_end": 1054, "temple_name": "Lý Phật Mã"},
            {"name": "Lý Thánh Tông", "reign_start": 1054, "reign_end": 1072, "description": "Đổi quốc hiệu thành Đại Việt."},
            {"name": "Lý Nhân Tông", "reign_start": 1072, "reign_end": 1127, "description": "Lý Thường Kiệt đánh Tống, giữ vững biên cương."},
            {"name": "Lý Thần Tông", "reign_start": 1128, "reign_end": 1138},
            {"name": "Lý Anh Tông", "reign_start": 1138, "reign_end": 1175},
            {"name": "Lý Cao Tông", "reign_start": 1176, "reign_end": 1210},
            {"name": "Lý Huệ Tông", "reign_start": 1210, "reign_end": 1224},
            {"name": "Lý Chiêu Hoàng", "reign_start": 1224, "reign_end": 1225, "description": "Nữ hoàng cuối cùng nhà Lý, nhường ngôi cho Trần Cảnh."},
        ],
    },
    {
        "slug": "nha-tran",
        "name": "Nhà Trần",
        "start_year": 1225,
        "end_year": 1400,
        "color": "#E74C3C",
        "description": "Ba lần đại thắng quân Nguyên Mông. Hưng Đạo Vương Trần Quốc Tuấn — anh hùng dân tộc vĩ đại.",
        "kings": [
            {"name": "Trần Thái Tông", "reign_start": 1225, "reign_end": 1258, "temple_name": "Trần Cảnh"},
            {"name": "Trần Thánh Tông", "reign_start": 1258, "reign_end": 1278},
            {"name": "Trần Nhân Tông", "reign_start": 1278, "reign_end": 1293,
             "description": "Lãnh đạo kháng chiến chống Nguyên Mông lần 2 và 3. Sáng lập Thiền phái Trúc Lâm."},
            {"name": "Trần Anh Tông", "reign_start": 1293, "reign_end": 1314},
            {"name": "Trần Minh Tông", "reign_start": 1314, "reign_end": 1329},
            {"name": "Trần Hiến Tông", "reign_start": 1329, "reign_end": 1341},
            {"name": "Trần Dụ Tông", "reign_start": 1341, "reign_end": 1369},
            {"name": "Trần Nghệ Tông", "reign_start": 1370, "reign_end": 1372},
            {"name": "Trần Duệ Tông", "reign_start": 1373, "reign_end": 1377},
            {"name": "Trần Phế Đế", "reign_start": 1377, "reign_end": 1388},
            {"name": "Trần Thuận Tông", "reign_start": 1388, "reign_end": 1398},
            {"name": "Trần Thiếu Đế", "reign_start": 1398, "reign_end": 1400},
        ],
    },
    {
        "slug": "nha-ho",
        "name": "Nhà Hồ",
        "start_year": 1400,
        "end_year": 1407,
        "color": "#1ABC9C",
        "description": "Hồ Quý Ly cải cách táo bạo. Nhanh chóng sụp đổ trước cuộc xâm lược của nhà Minh.",
        "kings": [
            {"name": "Hồ Quý Ly", "reign_start": 1400, "reign_end": 1401, "description": "Đổi quốc hiệu thành Đại Ngu. Cải cách tiền tệ, điền địa, giáo dục."},
            {"name": "Hồ Hán Thương", "reign_start": 1401, "reign_end": 1407, "description": "Thua quân Minh, kết thúc nhà Hồ."},
        ],
    },
    {
        "slug": "nha-le-so",
        "name": "Nhà Lê sơ",
        "start_year": 1428,
        "end_year": 1527,
        "color": "#2C3E50",
        "description": "Lê Lợi sau 10 năm kháng chiến giải phóng đất nước. Lê Thánh Tông — hoàng đế văn trị võ công lừng lẫy.",
        "kings": [
            {"name": "Lê Thái Tổ", "reign_start": 1428, "reign_end": 1433, "temple_name": "Lê Lợi",
             "description": "Lãnh đạo khởi nghĩa Lam Sơn 10 năm, đuổi quân Minh, lập nhà Lê."},
            {"name": "Lê Thái Tông", "reign_start": 1433, "reign_end": 1442},
            {"name": "Lê Nhân Tông", "reign_start": 1442, "reign_end": 1459},
            {"name": "Lê Thánh Tông", "reign_start": 1460, "reign_end": 1497,
             "description": "Biên soạn Hồng Đức Quốc Âm Thi Tập, mở rộng lãnh thổ vào phía Nam. Thời kỳ hoàng kim."},
            {"name": "Lê Hiến Tông", "reign_start": 1497, "reign_end": 1504},
            {"name": "Lê Túc Tông", "reign_start": 1504, "reign_end": 1504},
            {"name": "Lê Uy Mục", "reign_start": 1505, "reign_end": 1509},
            {"name": "Lê Tương Dực", "reign_start": 1510, "reign_end": 1516},
            {"name": "Lê Chiêu Tông", "reign_start": 1516, "reign_end": 1522},
            {"name": "Lê Cung Hoàng", "reign_start": 1522, "reign_end": 1527},
        ],
    },
    {
        "slug": "mac-le-trung-hung",
        "name": "Nhà Mạc & Lê Trung Hưng",
        "start_year": 1527,
        "end_year": 1788,
        "color": "#6C3483",
        "description": "Phân tranh Nam–Bắc triều. Trịnh–Nguyễn phân tranh chia đôi đất nước. Các chúa Nguyễn mở cõi phương Nam.",
        "kings": [
            {"name": "Mạc Đăng Dung", "reign_start": 1527, "reign_end": 1529, "description": "Lập nhà Mạc, gây ra cuộc chiến Nam–Bắc triều."},
            {"name": "Lê Trang Tông", "reign_start": 1533, "reign_end": 1548, "description": "Lê Trung Hưng, được Nguyễn Kim phò tá."},
            {"name": "Lê Trung Tông (Trung Hưng)", "reign_start": 1548, "reign_end": 1556},
            {"name": "Lê Anh Tông", "reign_start": 1556, "reign_end": 1573},
            {"name": "Lê Thế Tông", "reign_start": 1573, "reign_end": 1599},
            {"name": "Lê Kính Tông", "reign_start": 1600, "reign_end": 1619},
            {"name": "Lê Thần Tông", "reign_start": 1619, "reign_end": 1643},
            {"name": "Lê Chân Tông", "reign_start": 1643, "reign_end": 1649},
            {"name": "Lê Huyền Tông", "reign_start": 1663, "reign_end": 1671},
            {"name": "Lê Gia Tông", "reign_start": 1672, "reign_end": 1675},
            {"name": "Lê Hy Tông", "reign_start": 1676, "reign_end": 1705},
            {"name": "Lê Dụ Tông", "reign_start": 1705, "reign_end": 1729},
            {"name": "Lê Hiển Tông", "reign_start": 1740, "reign_end": 1786},
            {"name": "Lê Chiêu Thống", "reign_start": 1786, "reign_end": 1789, "description": "Vua cuối nhà Lê, cầu viện nhà Thanh, bị Quang Trung đánh tan."},
        ],
    },
    {
        "slug": "nha-tay-son",
        "name": "Nhà Tây Sơn",
        "start_year": 1778,
        "end_year": 1802,
        "color": "#E67E22",
        "description": "Phong trào nông dân vĩ đại. Quang Trung đại phá quân Thanh Tết Kỷ Dậu 1789 — chiến thắng quân sự lừng lẫy nhất.",
        "kings": [
            {"name": "Thái Đức (Nguyễn Nhạc)", "reign_start": 1778, "reign_end": 1793, "description": "Anh cả ba anh em Tây Sơn, xưng Hoàng đế."},
            {"name": "Quang Trung (Nguyễn Huệ)", "reign_start": 1788, "reign_end": 1792,
             "description": "Đại phá 29 vạn quân Thanh Tết Kỷ Dậu. Thiên tài quân sự kiệt xuất. Mất sớm năm 39 tuổi."},
            {"name": "Cảnh Thịnh (Nguyễn Quang Toản)", "reign_start": 1792, "reign_end": 1802, "description": "Con Quang Trung, thua Nguyễn Ánh, kết thúc nhà Tây Sơn."},
        ],
    },
    {
        "slug": "nha-nguyen",
        "name": "Nhà Nguyễn",
        "start_year": 1802,
        "end_year": 1945,
        "color": "#922B21",
        "description": "Triều đại phong kiến cuối cùng. Thống nhất toàn vẹn lãnh thổ. Thoái vị trước Cách mạng tháng Tám 1945.",
        "kings": [
            {"name": "Gia Long", "reign_start": 1802, "reign_end": 1820, "temple_name": "Nguyễn Ánh",
             "description": "Thống nhất đất nước sau nhiều thập kỷ chinh chiến. Đặt tên nước là Việt Nam."},
            {"name": "Minh Mạng", "reign_start": 1820, "reign_end": 1841, "description": "Vua cai trị tài giỏi, mở rộng bộ máy nhà nước."},
            {"name": "Thiệu Trị", "reign_start": 1841, "reign_end": 1847},
            {"name": "Tự Đức", "reign_start": 1847, "reign_end": 1883, "description": "Trị vì lâu nhất nhà Nguyễn. Đất nước rơi vào tay Pháp."},
            {"name": "Hiệp Hòa", "reign_start": 1883, "reign_end": 1883},
            {"name": "Kiến Phúc", "reign_start": 1883, "reign_end": 1884},
            {"name": "Hàm Nghi", "reign_start": 1884, "reign_end": 1885, "description": "Chiếu Cần Vương kháng Pháp, bị lưu đày Algeria."},
            {"name": "Đồng Khánh", "reign_start": 1885, "reign_end": 1889},
            {"name": "Thành Thái", "reign_start": 1889, "reign_end": 1907, "description": "Có tư tưởng chống Pháp, bị đày sang châu Phi."},
            {"name": "Duy Tân", "reign_start": 1907, "reign_end": 1916, "description": "Tham gia khởi nghĩa chống Pháp, bị đày."},
            {"name": "Khải Định", "reign_start": 1916, "reign_end": 1925},
            {"name": "Bảo Đại", "reign_start": 1926, "reign_end": 1945,
             "description": "Vua cuối cùng. Thoái vị ngày 25/8/1945 trước Cách mạng tháng Tám."},
        ],
    },
    {
        "slug": "viet-nam-hien-dai",
        "name": "Việt Nam Hiện Đại",
        "start_year": 1945,
        "end_year": None,
        "color": "#DA0000",
        "description": "Cách mạng tháng Tám 1945. Kháng chiến chống Pháp, chống Mỹ. Thống nhất đất nước 1975. Đổi Mới 1986.",
        "kings": [
            {"name": "Hồ Chí Minh", "reign_start": 1945, "reign_end": 1969, "temple_name": "Chủ tịch nước",
             "description": "Sáng lập nước VNDCCH, lãnh đạo kháng chiến chống Pháp và chống Mỹ. Cha già dân tộc."},
            {"name": "Tôn Đức Thắng", "reign_start": 1969, "reign_end": 1980, "temple_name": "Chủ tịch nước"},
            {"name": "Trường Chinh", "reign_start": 1981, "reign_end": 1987, "temple_name": "Chủ tịch Hội đồng Nhà nước",
             "description": "Khởi xướng chính sách Đổi Mới 1986."},
            {"name": "Võ Chí Công", "reign_start": 1987, "reign_end": 1992, "temple_name": "Chủ tịch Hội đồng Nhà nước"},
            {"name": "Lê Đức Anh", "reign_start": 1992, "reign_end": 1997, "temple_name": "Chủ tịch nước"},
            {"name": "Trần Đức Lương", "reign_start": 1997, "reign_end": 2006, "temple_name": "Chủ tịch nước"},
            {"name": "Nguyễn Minh Triết", "reign_start": 2006, "reign_end": 2011, "temple_name": "Chủ tịch nước"},
            {"name": "Trương Tấn Sang", "reign_start": 2011, "reign_end": 2016, "temple_name": "Chủ tịch nước"},
            {"name": "Trần Đại Quang", "reign_start": 2016, "reign_end": 2018, "temple_name": "Chủ tịch nước"},
            {"name": "Nguyễn Phú Trọng", "reign_start": 2018, "reign_end": 2024, "temple_name": "Chủ tịch nước",
             "description": "Kiêm Tổng Bí thư, đẩy mạnh chống tham nhũng."},
            {"name": "Lương Cường", "reign_start": 2024, "reign_end": 2026, "temple_name": "Chủ tịch nước"},
            {"name": "Tô Lâm", "reign_start": 2026, "reign_end": None, "temple_name": "Chủ tịch nước"},
        ],
    },
]

# ── Insert ────────────────────────────────────────────────────────────────────
for i, d in enumerate(DYNASTIES):
    kings_data = d.pop("kings", [])
    dynasty = Dynasty(
        slug=d["slug"],
        name=d["name"],
        start_year=d["start_year"],
        end_year=d.get("end_year"),
        description=d.get("description", ""),
        color=d["color"],
        order=i,
    )
    dynasty.save(using=DB)

    for j, k in enumerate(kings_data):
        King(
            dynasty=dynasty,
            name=k["name"],
            reign_start=k.get("reign_start"),
            reign_end=k.get("reign_end"),
            temple_name=k.get("temple_name", ""),
            description=k.get("description", ""),
            order=j,
        ).save(using=DB)

    print(f"[OK] {dynasty.name} — {len(kings_data)} người")

total_d = Dynasty.objects.using(DB).count()
total_k = King.objects.using(DB).count()
print(f"\n✅ Seed hoàn tất: {total_d} triều đại, {total_k} nhân vật/vua")
