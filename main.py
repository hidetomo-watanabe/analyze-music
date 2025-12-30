import librosa
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Section:
    """楽曲のセクション情報"""
    name: str
    start_bar: int
    end_bar: int
    start_time: float
    end_time: float
    chords: List[str] = None  # コード進行を追加

    @property
    def duration_bars(self):
        return self.end_bar - self.start_bar + 1

    def __str__(self):
        base = f"{self.name:12} | 小節 {self.start_bar:3d}-{self.end_bar:3d} ({self.duration_bars:2d}小節) | {self.start_time:6.2f}s - {self.end_time:6.2f}s"
        if self.chords:
            chord_str = " -> ".join(self.chords[:8])  # 最初の8コードまで表示
            if len(self.chords) > 8:
                chord_str += "..."
            base += f"\n             コード: {chord_str}"
        return base

class MusicStructureAnalyzer:
    """音源から楽曲構成を分析"""

    def __init__(self, audio_path: str, bpm: float = None, beats_per_bar: int = 4):
        """
        Parameters:
        - audio_path: 音源ファイルのパス
        - bpm: BPM（指定しない場合は自動推定）
        - beats_per_bar: 1小節の拍数（デフォルト4拍子）
        """
        print(f"音源を読み込み中: {audio_path}")
        self.y, self.sr = librosa.load(audio_path)
        self.beats_per_bar = beats_per_bar

        # BPM推定
        if bpm is None:
            tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
            # numpy配列からスカラー値を安全に取得
            self.bpm = float(tempo.item() if hasattr(tempo, 'item') else tempo)
            print(f"推定BPM: {self.bpm:.1f}")
        else:
            self.bpm = bpm
            print(f"指定BPM: {self.bpm}")

        # ビート検出
        self.tempo, self.beat_frames = librosa.beat.beat_track(
            y=self.y, sr=self.sr, bpm=self.bpm
        )
        self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)

        # コード検出用のクロマ特徴量を計算
        self.chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)

        # 小節の計算
        self.total_beats = len(self.beat_times)
        self.total_bars = self.total_beats // beats_per_bar
        self.duration = len(self.y) / self.sr

        print(f"総ビート数: {self.total_beats}")
        print(f"総小節数: {self.total_bars}")
        print(f"曲の長さ: {self.duration:.2f}秒")

    def time_to_bar(self, time_sec: float) -> int:
        """時間（秒）を小節番号に変換"""
        beat_idx = np.searchsorted(self.beat_times, time_sec)
        bar_num = (beat_idx // self.beats_per_bar) + 1
        return min(bar_num, self.total_bars)

    def bar_to_time(self, bar_num: int) -> float:
        """小節番号を時間（秒）に変換"""
        beat_idx = (bar_num - 1) * self.beats_per_bar
        if beat_idx < len(self.beat_times):
            return self.beat_times[beat_idx]
        return self.duration

    def detect_chords(self, start_time: float, end_time: float) -> List[str]:
        """指定時間範囲のコード進行を検出"""
        # コード名のマッピング
        chord_templates = {
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'C#': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            'D': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            'D#': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            'E': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'F#': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'G#': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            'A': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            'A#': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            'B': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        }

        # 時間をフレームに変換
        start_frame = librosa.time_to_frames(start_time, sr=self.sr)
        end_frame = librosa.time_to_frames(end_time, sr=self.sr)

        # 該当範囲のクロマ特徴量を取得
        chroma_section = self.chroma[:, start_frame:end_frame]

        # 小節ごとにコードを検出
        section_duration = end_time - start_time
        bars_in_section = int((end_frame - start_frame) / (chroma_section.shape[1] / (section_duration * self.bpm / 60 / self.beats_per_bar)))
        bars_in_section = max(1, min(bars_in_section, 32))  # 1〜32小節に制限

        chords = []
        frames_per_bar = max(1, chroma_section.shape[1] // bars_in_section)

        for i in range(bars_in_section):
            start_idx = i * frames_per_bar
            end_idx = min((i + 1) * frames_per_bar, chroma_section.shape[1])

            if start_idx >= chroma_section.shape[1]:
                break

            # その小節のクロマを平均
            bar_chroma = np.mean(chroma_section[:, start_idx:end_idx], axis=1)

            # 最も近いコードを見つける
            best_chord = 'N/A'
            best_score = -1

            for chord_name, template in chord_templates.items():
                score = np.dot(bar_chroma, template)
                if score > best_score:
                    best_score = score
                    best_chord = chord_name

            chords.append(best_chord)

        return chords

    def detect_sections_auto(self, detect_chords: bool = True) -> List[Section]:
        """自動でセクション（イントロ、Aメロ等）を検出"""
        # セグメント境界を検出（音響特徴の変化点）
        bounds = librosa.segment.agglomerative(self.chroma, 8)
        bound_times = librosa.frames_to_time(bounds, sr=self.sr)

        sections = []
        section_names = ["Intro", "A", "B", "Chorus", "Bridge", "C", "Outro"]

        for i, (start, end) in enumerate(zip(bound_times[:-1], bound_times[1:])):
            name = section_names[i] if i < len(section_names) else f"Section{i+1}"
            start_bar = self.time_to_bar(start)
            end_bar = self.time_to_bar(end) - 1

            # コード進行を検出
            chords = None
            if detect_chords:
                chords = self.detect_chords(start, end)

            sections.append(Section(
                name=name,
                start_bar=start_bar,
                end_bar=end_bar,
                start_time=start,
                end_time=end,
                chords=chords
            ))

        return sections

    def create_manual_sections(self, section_list: List[tuple], detect_chords: bool = True) -> List[Section]:
        """
        手動でセクションを作成

        Parameters:
        - section_list: [(名前, 開始小節, 終了小節), ...] のリスト
        - detect_chords: コード進行を検出するか

        例: [("Intro", 1, 8), ("Aメロ", 9, 24), ("サビ", 25, 40)]
        """
        sections = []
        for name, start_bar, end_bar in section_list:
            start_time = self.bar_to_time(start_bar)
            end_time = self.bar_to_time(end_bar + 1)

            # コード進行を検出
            chords = None
            if detect_chords:
                chords = self.detect_chords(start_time, end_time)

            sections.append(Section(
                name=name,
                start_bar=start_bar,
                end_bar=end_bar,
                start_time=start_time,
                end_time=end_time,
                chords=chords
            ))

        return sections

    def print_structure(self, sections: List[Section], output_file: str = None):
        """楽曲構成を表示・保存"""
        output = []
        output.append("=" * 70)
        output.append("楽曲構成")
        output.append("=" * 70)
        output.append(f"BPM: {self.bpm:.1f} | 拍子: {self.beats_per_bar}/4 | 総小節数: {self.total_bars}")
        output.append("-" * 70)

        for section in sections:
            output.append(str(section))

        output.append("=" * 70)
        output.append(f"合計: {sum(s.duration_bars for s in sections)}小節")

        # コンソールに表示
        print("\n" + "\n".join(output))

        # ファイルに保存
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(output))
            print(f"\n構成をテキストファイルに保存: {output_file}")

    def visualize_structure(self, sections: List[Section], output_path: str = "structure.png"):
        """楽曲構成を可視化（この機能は無効化されています）"""
        # PNG出力は文字化けするため無効化
        pass


# 使用例
if __name__ == "__main__":
    import argparse

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='音源から楽曲構成を分析')
    parser.add_argument('audio_file', help='音源ファイルのパス (WAV, MP3, M4A等)')
    parser.add_argument('--bpm', type=float, default=None,
                       help='BPMを指定（省略時は自動推定）')
    parser.add_argument('--beats', type=int, default=4,
                       help='1小節の拍数（デフォルト: 4）')
    parser.add_argument('--auto-only', action='store_true',
                       help='自動検出のみ実行')
    parser.add_argument('--output', type=str, default=None,
                       help='出力ファイル名のプレフィックス')
    parser.add_argument('--no-chords', action='store_true',
                       help='コード進行の検出を無効化')

    args = parser.parse_args()
    audio_file = args.audio_file

    # outputディレクトリを作成
    import os
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 出力ファイル名の決定
    if args.output:
        output_prefix = args.output
    else:
        output_prefix = os.path.splitext(os.path.basename(audio_file))[0]

    # 出力パスを output/ 配下に設定
    output_path = os.path.join(output_dir, f"{output_prefix}_auto.txt")

    try:
        # 分析器を初期化
        analyzer = MusicStructureAnalyzer(audio_file, bpm=args.bpm, beats_per_bar=args.beats)

        # 自動検出
        print("\n【自動検出】")
        auto_sections = analyzer.detect_sections_auto(detect_chords=not args.no_chords)
        analyzer.print_structure(auto_sections, output_path)

        # 手動設定の例（--auto-onlyが指定されていない場合）
        if not args.auto_only:
            print("\n【手動設定の例】")
            print("以下のように手動で構成を設定できます：")
            print("manual_sections = analyzer.create_manual_sections([")
            print("    ('Intro', 1, 8),")
            print("    ('Aメロ', 9, 24),")
            print("    ('サビ', 25, 40),")
            print("])")
            print("analyzer.print_structure(manual_sections, 'output/manual.txt')")
            print("\nコード内で編集してください。")

    except FileNotFoundError:
        print(f"\nエラー: '{audio_file}' が見つかりません")
        print("音源ファイルのパスを確認してください")
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()
