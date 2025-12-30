import librosa
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import List
import os

@dataclass
class Bar:
    """小節情報"""
    bar_num: int
    start_time: float
    end_time: float
    chord: str
    avg_intensity: float  # 平均音量

    def __str__(self):
        intensity_bar = "■" * int(self.avg_intensity * 20)
        return f"小節 {self.bar_num:3d} | {self.start_time:6.2f}s - {self.end_time:6.2f}s | コード: {self.chord:4s} | 強度: {intensity_bar}"

class DrumCopyHelper:
    """ドラム耳コピ支援ツール"""

    def __init__(self, audio_path: str, bpm: float = None, beats_per_bar: int = 4, skip_initial_silence: bool = True):
        print(f"音源を読み込み中: {audio_path}")
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path)
        self.beats_per_bar = beats_per_bar
        self.audio_start_time = 0.0

        # 冒頭の無音をスキップ
        if skip_initial_silence:
            non_silent_intervals = librosa.effects.split(self.y, top_db=40)
            if len(non_silent_intervals) > 0:
                start_sample = non_silent_intervals[0][0]
                self.audio_start_time = librosa.samples_to_time(start_sample, sr=self.sr)
                if self.audio_start_time > 0.5:
                    print(f"冒頭の無音をスキップ: {self.audio_start_time:.2f}秒")

        # BPM推定
        if bpm is None:
            tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
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

        # 音声開始位置でビートを調整
        if skip_initial_silence and self.audio_start_time > 0:
            self.beat_times = self.beat_times[self.beat_times >= self.audio_start_time]

        # 小節数を計算
        self.total_beats = len(self.beat_times)
        self.total_bars = self.total_beats // beats_per_bar
        self.duration = len(self.y) / self.sr

        # クロマ特徴量を計算（コード検出用）
        self.chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)

        # ドラム音源を分離
        print("ドラムパートを分離中...")
        self.D = librosa.stft(self.y)
        self.D_harmonic, self.D_percussive = librosa.decompose.hpss(self.D, margin=3.0)
        self.y_drums = librosa.istft(self.D_percussive)
        self.y_no_drums = librosa.istft(self.D_harmonic)

        # RMS（音量）を計算
        self.rms = librosa.feature.rms(y=self.y_drums)[0]

        print(f"総小節数: {self.total_bars}")

    def detect_chord(self, start_time: float, end_time: float) -> str:
        """指定時間範囲のコードを検出"""
        chord_templates = {
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'C#': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            'D': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            'Eb': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            'E': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'F#': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Ab': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            'A': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            'Bb': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            'B': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        }

        start_frame = librosa.time_to_frames(start_time, sr=self.sr)
        end_frame = librosa.time_to_frames(end_time, sr=self.sr)

        if start_frame >= self.chroma.shape[1]:
            return '-'

        end_frame = min(end_frame, self.chroma.shape[1])
        bar_chroma = np.mean(self.chroma[:, start_frame:end_frame], axis=1)

        best_chord = '-'
        best_score = -1

        for chord_name, template in chord_templates.items():
            score = np.dot(bar_chroma, template)
            if score > best_score:
                best_score = score
                best_chord = chord_name

        return best_chord

    def analyze_bars(self) -> List[Bar]:
        """全小節を分析"""
        bars = []

        for bar_num in range(1, self.total_bars + 1):
            start_beat_idx = (bar_num - 1) * self.beats_per_bar
            end_beat_idx = bar_num * self.beats_per_bar

            if start_beat_idx >= len(self.beat_times):
                break

            start_time = self.beat_times[start_beat_idx]
            end_time = self.beat_times[end_beat_idx] if end_beat_idx < len(self.beat_times) else self.duration

            # コード検出
            chord = self.detect_chord(start_time, end_time)

            # 音量（強度）を計算
            start_frame = librosa.time_to_frames(start_time, sr=self.sr, hop_length=512)
            end_frame = librosa.time_to_frames(end_time, sr=self.sr, hop_length=512)
            end_frame = min(end_frame, len(self.rms))

            if start_frame < len(self.rms):
                avg_intensity = np.mean(self.rms[start_frame:end_frame])
            else:
                avg_intensity = 0

            bars.append(Bar(
                bar_num=bar_num,
                start_time=start_time,
                end_time=end_time,
                chord=chord,
                avg_intensity=avg_intensity
            ))

        return bars

    def print_bars_with_intensity(self, bars: List[Bar], output_file: str = None):
        """小節情報とビート強度を統合して表示・保存"""
        output = []
        output.append("=" * 90)
        output.append("ドラム耳コピ支援")
        output.append("=" * 90)
        output.append(f"BPM: {self.bpm:.1f} | 拍子: {self.beats_per_bar}/4 | 総小節数: {len(bars)}")
        output.append("=" * 90)

        # 強度マップ（コードと秒数付き）
        output.append("\n【ドラムの強度マップ】")
        output.append("-" * 90)

        # 8小節ごとにグループ化
        for i in range(0, len(bars), 8):
            group = bars[i:i+8]
            output.append(f"\n小節 {group[0].bar_num:3d} - {group[-1].bar_num:3d}:")

            for bar in group:
                intensity_bar = "█" * int(bar.avg_intensity * 40)
                output.append(f"  {bar.bar_num:3d} | {intensity_bar:40s} | {bar.chord:4s} | {bar.start_time:6.2f}s - {bar.end_time:6.2f}s")

        output.append("\n" + "=" * 90)

        # コンソールに表示
        print("\n" + "\n".join(output))

        # ファイルに保存
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(output))
            print(f"\n結果を保存: {output_file}")

    def export_drums_only(self, output_path: str):
        """ドラムだけを抽出した音源を生成"""
        print(f"\nドラムのみの音源を生成中...")
        sf.write(output_path, self.y_drums, self.sr)
        print(f"保存完了: {output_path}")

    def export_no_drums(self, output_path: str):
        """ドラムを除去した音源を生成"""
        print(f"\nドラム除去音源を生成中...")
        sf.write(output_path, self.y_no_drums, self.sr)
        print(f"保存完了: {output_path}")

    def export_slowed(self, output_path: str, speed: float = 0.75):
        """テンポを落とした音源を生成"""
        print(f"\nテンポを{int(speed*100)}%に変更中...")
        y_slow = librosa.effects.time_stretch(self.y, rate=speed)
        sf.write(output_path, y_slow, self.sr)
        print(f"保存完了: {output_path}")

    def export_bar_range(self, output_path: str, start_bar: int, end_bar: int, bars: List[Bar]):
        """特定の小節範囲を切り出し"""
        if start_bar < 1 or end_bar > len(bars):
            print(f"エラー: 小節範囲が不正です（1-{len(bars)}の範囲で指定してください）")
            return

        start_time = bars[start_bar - 1].start_time
        end_time = bars[end_bar - 1].end_time

        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        y_excerpt = self.y[start_sample:end_sample]

        print(f"\n小節{start_bar}〜{end_bar}を切り出し中...")
        sf.write(output_path, y_excerpt, self.sr)
        print(f"保存完了: {output_path}")

    def visualize_intensity(self, output_path: str, bars: List[Bar]):
        """ビートの強弱を可視化（この機能は print_bars_with_intensity に統合されました）"""
        pass


# メイン実行
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ドラム耳コピ支援ツール')
    parser.add_argument('audio_file', help='音源ファイルのパス (WAV, MP3等)')
    parser.add_argument('--bpm', type=float, default=None, help='BPMを指定（省略時は自動推定）')
    parser.add_argument('--beats', type=int, default=4, help='1小節の拍数（デフォルト: 4）')
    parser.add_argument('--no-skip-silence', action='store_true', help='冒頭の無音をスキップしない')
    parser.add_argument('--output', type=str, default=None, help='出力ファイル名のプレフィックス')

    # 機能オプション
    parser.add_argument('--drums-only', action='store_true', help='ドラムのみの音源を生成')
    parser.add_argument('--no-drums', action='store_true', help='ドラム除去音源を生成')
    parser.add_argument('--slow', type=float, default=None, help='テンポを落とした音源を生成（例: 0.75で75%）')
    parser.add_argument('--extract', type=str, default=None, help='特定の小節を切り出し（例: "25-32"）')
    parser.add_argument('--all', action='store_true', help='すべての音源生成機能を実行')

    args = parser.parse_args()
    audio_file = args.audio_file

    # outputディレクトリを作成
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 出力ファイル名の決定
    if args.output:
        output_prefix = args.output
    else:
        output_prefix = os.path.splitext(os.path.basename(audio_file))[0]

    try:
        # 分析器を初期化
        helper = DrumCopyHelper(
            audio_file,
            bpm=args.bpm,
            beats_per_bar=args.beats,
            skip_initial_silence=not args.no_skip_silence
        )

        # 小節を分析
        print("\n小節ごとに分析中...")
        bars = helper.analyze_bars()

        # 結果を表示・保存（小節一覧 + 強度マップを統合）
        output_file = os.path.join(output_dir, f"{output_prefix}_analysis.txt")
        helper.print_bars_with_intensity(bars, output_file)

        # 各機能の実行
        if args.all or args.drums_only:
            drums_output = os.path.join(output_dir, f"{output_prefix}_drums_only.wav")
            helper.export_drums_only(drums_output)

        if args.all or args.no_drums:
            no_drums_output = os.path.join(output_dir, f"{output_prefix}_no_drums.wav")
            helper.export_no_drums(no_drums_output)

        if args.all or args.slow:
            speed = args.slow if args.slow else 0.75
            slow_output = os.path.join(output_dir, f"{output_prefix}_slow_{int(speed*100)}.wav")
            helper.export_slowed(slow_output, speed)

        if args.extract:
            try:
                start_bar, end_bar = map(int, args.extract.split('-'))
                extract_output = os.path.join(output_dir, f"{output_prefix}_bars_{start_bar}-{end_bar}.wav")
                helper.export_bar_range(extract_output, start_bar, end_bar, bars)
            except ValueError:
                print("エラー: --extract のフォーマットが不正です（例: --extract 25-32）")

        print("\n" + "=" * 90)
        print("完了！")
        print("=" * 90)

    except FileNotFoundError:
        print(f"\nエラー: '{audio_file}' が見つかりません")
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()
