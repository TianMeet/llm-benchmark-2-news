"""
选股宝板块数据客户端

获取选股宝全部板块数据并存储到 SQLite。
不依赖 src.config，可独立使用。

表结构:
  plates              - 板块基本信息与行情统计
  stocks              - 板块成分股（含核心标记、产业链归属、入选理由）
  industrial_chains   - 板块产业链环节
  question_answers    - 板块介绍问答（HTML富文本）
"""

import logging
import random
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# 完整板块ID列表（来源: 选股宝板块列表接口）
ALL_PLATE_IDS = [
    30553810, 86700521, 61407774, 6784718, 16940137, 4670889, 32014734, 58346049,
    24175361, 19218721, 36001721, 54463873, 63576990, 12203929, 44167378, 47059342,
    96955826, 60711310, 18608521, 93009017, 1822777, 35232754, 20212110, 17798814,
    52532754, 24898553, 32942745, 93064114, 50522441, 30712462, 91348766, 65767826,
    88981697, 17330610, 16844702, 13492121, 19022049, 19384882, 98292897, 62132338,
    89343809, 22082034, 85637417, 16856569, 95628809, 14183966, 19993841, 50285650,
    124498, 29073617, 61549817, 1277665, 18811806, 88130137, 18311442, 18580382,
    60280094, 87009566, 20544889, 28126622, 21008649, 17277266, 26543321, 19939217,
    19088937, 17412529, 651982, 17957649, 16845721, 17752718, 65470862, 97910113,
    63722137, 47530770, 89938137, 35562385, 49246110, 32686798, 52153249, 90484942,
    31163550, 16988830, 66663630, 17731137, 19461278, 38844242, 26810001, 27772369,
    26346142, 90434033, 14791054, 36776334, 35079641, 25513273, 16238750, 85990881,
    20422674, 27430942, 19522290, 45837518, 30370802, 85460510, 87592142, 12679250,
    34165697, 42880657, 26881822, 18533889, 5464089, 3471566, 96634065, 44702162,
    36334226, 19426649, 66383121, 97396649, 84537970, 41542162, 20876942, 97970574,
    17548129, 53944073, 20747785, 16827022, 14421778, 31765737, 55118706, 22361929,
    45334105, 18210194, 54203009, 22665673, 91769249, 85063761, 42848050, 24409497,
    18722817, 87902321, 20054814, 27711182, 41087006, 46043186, 21095762, 29448210,
    16858654, 16888094, 17290881, 15621326, 17640222, 42683577, 49765121, 84888270,
    13960785, 18801358, 56044969, 22585502, 59173721, 66814321, 45538417, 59034782,
    95252353, 58896585, 94313886, 17032337, 65619233, 49981185, 17236249, 18693774,
    26944697, 23848497, 29751954, 15742577, 90798737, 19627721, 52405857, 44002121,
    38685193, 50645662, 6860722, 22180510, 48425678, 20263570, 21317297, 19121426,
    20668110, 16793689, 15030162, 30952137, 25764977, 62275737, 60993810, 18366257,
    66232882, 92638377, 17347449, 16950418, 33504910, 45370610, 37556601, 6888401,
    40734161, 22512270, 51900521, 43341297, 48254033, 39480658, 30137233, 21553249,
    20454834, 100611657, 36931937, 20303009, 20922194, 17678642, 4966834, 67446849,
    47229409, 29982649, 42030497, 85108402, 6963833, 16843009, 16806857, 21969081,
    60429714, 25570510, 16861993, 65191137, 58072137, 93995801, 16930590, 16851537,
    19351097, 87544337, 88440990, 787705, 60139577, 63271937, 54594386, 38499865,
    17490281, 35716402, 39451777, 20372510, 30605897, 99898354, 44869065, 21051521,
    95687070, 49938062, 46550505, 54333070, 15500305, 18505298, 89575241, 2469022,
    21790494, 17864537, 387225, 5967502, 4245902, 51267729, 48213682, 31113561,
    20965249, 36156190, 17428174, 17174158, 19825650, 21506578, 25892441, 20332457,
    30214350, 18005049, 51018737, 42193042, 33200722, 96575182, 51518137, 93621937,
    16868321, 44202817, 97015161, 19977522, 17980978, 21367929, 16997081, 62982657,
    17567121, 96196466, 64895529, 51644190, 92268641, 16842834, 99510478, 12795137,
    61977294, 37087250, 23773810, 26021321, 19288337, 13011026, 16787698, 64014641,
    26150081, 31979922, 21183902, 17234334, 42225257, 25638417, 64879134, 29774622,
    16920249, 86475809, 24190721, 19488113, 18231838, 17529874, 53685529, 66964210,
    6039890, 6346009, 18994354, 33160094, 17313489, 432945, 40091417, 91715730,
    11974705, 17841310, 94501129, 62707282, 94875777, 4278945, 18551961, 33246089,
    85945106, 54856094, 23112594, 18771249, 17949897, 53427377, 21228313, 17249694,
    25195410, 26676721, 29393249, 21277137, 85415929, 44499598, 16849614, 63127182,
    59860753, 41736590, 17928146, 14909985, 24949522, 18415794, 55381710, 47701289,
    52659881, 21413918, 93566670, 7302825, 41380009, 16813522, 16961441, 89031310,
    23569950, 3185625, 16361417, 32479774, 44666049, 51142862, 1008158, 39292558,
    21453746, 53171153, 95309650, 5364594, 19251602, 23297057, 19919321, 16781937,
    25388521, 62419870, 16875041, 89393362, 39931294, 17380377, 24364114, 41573473,
    17606385, 17452705, 34795166, 16883689, 36489377, 65934622, 18336809, 37869777,
    91164857, 62563721, 23887774, 20093801, 28210305, 17819721, 59582321, 27217313,
    18064590, 17778649, 42389278, 59999794, 13843150, 1967310, 7198750, 55779193,
    16972174, 18443089, 17006066, 16116825, 93940082, 27292178, 37268510, 57930713,
    35408078, 40572622, 5433758, 55646642, 14303017, 698546, 48942489, 88490441,
    55250353, 32772401, 35385585, 16907470, 17091954, 48728625, 19771457, 52279694,
    17023182, 99835121, 84058014, 87950578, 38053809, 92084622, 17159801, 99062713,
    20836766, 2525138, 18469582, 29600142, 19881630, 86167950, 27912881, 13244318,
    87056697, 33731273, 5765970, 1875729, 22032881, 92322322, 37713330, 91400809,
    27631737, 97718290, 36179817, 18621010, 29544609, 61692082, 2900690, 22411474,
    98738974, 64438609, 17014769, 17065678, 90251762, 4637726, 48769650, 34642505,
    18651017, 15270174, 43506162, 14671841, 48043001, 17078449, 27571122, 16819905,
    17906113, 22768217, 17204050, 3506929, 16834881, 34468929, 43539673, 21362382,
    3857182, 966697, 55911454, 60851937, 32988274, 16892729, 17363998, 16841938,
    89887390, 90851122, 1914418, 23406049, 17512873, 37895886, 19532754, 95875609,
    22614482, 5937793, 1237390, 30764097, 5065425, 19713609, 22017842, 17053649,
    18926110, 18184937, 46512414, 23460626, 23962009, 17113697, 17709778, 46343481,
    26083982, 25072014, 24706738, 18160926, 16846450, 62120753, 40120690, 25019689,
    13608562, 21934737, 87365682, 6681777, 47740446, 20621170, 20793881, 99124530,
    18285281, 3126345, 94557522, 29828969, 17396594, 40280753, 94932622, 47020065,
    37112745, 21825682, 18258169, 16911154, 18393202, 20031694, 54724617, 16843401,
    84014217, 18041713, 18129294, 100286110, 46719890, 16981321, 21731726, 93250001,
    18963793, 19676882, 22331201, 23680690, 16483794, 67120850, 39640329, 741201,
    16847921, 19568577, 17627122, 38658462, 100223449, 67510926, 20495569, 92692510,
    84363673, 22114510, 1640169, 24291465, 98353810,
]


class XuangubaoClient:
    """选股宝板块数据客户端"""

    API_URL = "https://flash-api.xuangubao.com.cn/api/plate/plate_set?id={}"

    # SQLite 建表语句
    _CREATE_TABLE_SQLS = [
        '''CREATE TABLE IF NOT EXISTS plates (
            id                  INTEGER PRIMARY KEY,
            name                TEXT NOT NULL,
            desc                TEXT,
            subject_id          INTEGER,
            create_name         TEXT,
            update_name         TEXT,
            avg_pcp             REAL,
            core_avg_pcp        REAL,
            rise_count          INTEGER,
            fall_count          INTEGER,
            stay_count          INTEGER,
            fund_flow           REAL,
            has_north           INTEGER,
            core_stocks_count   INTEGER,
            hang_ye_long_tou_stocks_count INTEGER,
            updated_at          TEXT
        )''',
        '''CREATE TABLE IF NOT EXISTS stocks (
            plate_id            INTEGER NOT NULL,
            symbol              TEXT NOT NULL,
            desc                TEXT,
            desc_url            TEXT,
            industrial_chain_id INTEGER,
            is_new              INTEGER,
            need_pay            INTEGER,
            core_flag           INTEGER,
            PRIMARY KEY (plate_id, symbol)
        )''',
        '''CREATE TABLE IF NOT EXISTS industrial_chains (
            plate_id            INTEGER NOT NULL,
            chain_id            INTEGER NOT NULL,
            name                TEXT NOT NULL,
            "order"             INTEGER,
            PRIMARY KEY (plate_id, chain_id)
        )''',
        '''CREATE TABLE IF NOT EXISTS question_answers (
            plate_id            INTEGER NOT NULL,
            question            TEXT NOT NULL,
            html_answer         TEXT,
            PRIMARY KEY (plate_id, question)
        )''',
    ]

    def __init__(self, db_path: Optional[str] = None):
        """
        Parameters
        ----------
        db_path : str, optional
            SQLite 数据库路径。如不提供，则只能做 API 查询不能存储。
        """
        self.db_path = db_path

    def fetch_plate(self, plate_id: int) -> dict:
        """
        获取单个板块数据（API 调用）。

        Parameters
        ----------
        plate_id : int
            板块 ID

        Returns
        -------
        dict
            API 返回的 JSON 数据
        """
        resp = requests.get(self.API_URL.format(plate_id), timeout=15)
        resp.raise_for_status()
        return resp.json()

    def fetch_all_plates(self,
                         plate_ids: Optional[List[int]] = None,
                         interval: float = 3.0) -> dict:
        """
        批量获取板块数据，支持断点续爬。

        Parameters
        ----------
        plate_ids : list[int], optional
            板块 ID 列表，默认使用 ALL_PLATE_IDS
        interval : float
            基础请求间隔秒数（实际会随机波动为 0.8~2.5 倍）

        Returns
        -------
        dict
            {'success': int, 'fail': int, 'fail_ids': list}
        """
        if plate_ids is None:
            plate_ids = ALL_PLATE_IDS

        if self.db_path:
            self._init_db()

        # 断点续爬：跳过已成功爬取的板块
        done_ids = set()
        if self.db_path:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT id FROM plates')
            done_ids = {row[0] for row in c.fetchall()}
            conn.close()

        todo = [pid for pid in plate_ids if pid not in done_ids]
        total = len(plate_ids)
        skipped = total - len(todo)
        if skipped:
            logger.info(f"已完成 {skipped} 个板块，跳过；剩余 {len(todo)} 个待爬取")

        if not todo:
            logger.info("全部板块已爬取完毕")
            return {'success': 0, 'fail': 0, 'fail_ids': []}

        success = 0
        fail = 0
        fail_ids = []

        for i, pid in enumerate(todo, 1):
            try:
                data = self.fetch_plate(pid)
                if self.db_path:
                    self._save_plate(data)
                plate_name = data['data'].get('name', '未知')
                stock_count = len(data['data'].get('stocks', []))
                logger.info(f"[{skipped + i}/{total}] {plate_name}(id={pid}) 成分股={stock_count}")
                success += 1
            except Exception as e:
                logger.error(f"[{skipped + i}/{total}] 失败 id={pid}: {e}")
                fail += 1
                fail_ids.append(pid)

            if i < len(todo):
                sleep_time = interval * random.uniform(0.8, 2.5)
                if i % 30 == 0:
                    sleep_time += random.uniform(15, 30)
                time.sleep(sleep_time)

        return {'success': success, 'fail': fail, 'fail_ids': fail_ids}

    def get_plate_stocks(self, plate_id: int) -> List[dict]:
        """
        获取板块成分股列表。

        Parameters
        ----------
        plate_id : int
            板块 ID

        Returns
        -------
        list[dict]
            成分股列表，每项包含 symbol, desc, core_flag 等字段
        """
        data = self.fetch_plate(plate_id)
        if data.get('code') != 20000:
            raise Exception(f"API返回错误: {data.get('message')}")

        stocks = []
        for s in data['data'].get('stocks', []):
            stocks.append({
                'symbol': s['symbol'].split('.')[0],
                'desc': s.get('desc', ''),
                'desc_url': s.get('desc_url', ''),
                'industrial_chain_id': s.get('industrial_chain_id'),
                'is_new': bool(s.get('is_new')),
                'need_pay': bool(s.get('need_pay')),
                'core_flag': s.get('core_flag'),
            })
        return stocks

    def save_to_sqlite(self, db_path: Optional[str] = None):
        """
        将已爬取的数据保存到 SQLite。

        Parameters
        ----------
        db_path : str, optional
            如提供，覆盖实例的 db_path
        """
        if db_path:
            self.db_path = db_path
        if not self.db_path:
            raise ValueError("未指定 db_path")
        self._init_db()

    # ---- 内部方法 ----

    def _init_db(self):
        """初始化数据库，创建全部表"""
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for sql in self._CREATE_TABLE_SQLS:
            c.execute(sql)
        conn.commit()
        conn.close()

    def _save_plate(self, data: dict) -> int:
        """将一个板块的全部数据写入数据库，返回成分股数量"""
        if data['code'] != 20000:
            raise Exception(f"API返回错误: {data['message']}")

        d = data['data']
        plate_id = d['id']
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # plates
        c.execute('''INSERT OR REPLACE INTO plates VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
            plate_id,
            d.get('name', ''),
            d.get('desc', ''),
            d.get('subject_id'),
            d.get('create_name', ''),
            d.get('update_name', ''),
            d.get('avg_pcp'),
            d.get('core_avg_pcp'),
            d.get('rise_count'),
            d.get('fall_count'),
            d.get('stay_count'),
            d.get('fund_flow'),
            d.get('has_north'),
            d.get('core_stocks_count'),
            d.get('hang_ye_long_tou_stocks_count'),
            now,
        ))

        # stocks
        c.execute('DELETE FROM stocks WHERE plate_id = ?', (plate_id,))
        for s in d.get('stocks', []):
            c.execute('''INSERT INTO stocks VALUES (?,?,?,?,?,?,?,?)''', (
                plate_id,
                s['symbol'].split('.')[0],
                s.get('desc', ''),
                s.get('desc_url', ''),
                s.get('industrial_chain_id'),
                1 if s.get('is_new') else 0,
                1 if s.get('need_pay') else 0,
                s.get('core_flag'),
            ))

        # industrial_chains
        c.execute('DELETE FROM industrial_chains WHERE plate_id = ?', (plate_id,))
        for chain in d.get('industrial_chains') or []:
            c.execute('''INSERT INTO industrial_chains VALUES (?,?,?,?)''', (
                plate_id,
                chain['id'],
                chain['name'],
                chain.get('order'),
            ))

        # question_answers
        c.execute('DELETE FROM question_answers WHERE plate_id = ?', (plate_id,))
        for qa in d.get('question_answers') or []:
            c.execute('''INSERT INTO question_answers VALUES (?,?,?)''', (
                plate_id,
                qa['question'],
                qa.get('html_answer', ''),
            ))

        conn.commit()
        conn.close()
        return len(d.get('stocks', []))
