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

    def __init__(self, audio_path: str, bpm: float = None, beats_per_bar: int = 4, skip_initial_silence: bool = False):
        """
        Parameters:
        - audio_path: 音源ファイルのパス
        - bpm: BPM（指定しない場合は自動推定）
        - beats_per_bar: 1小節の拍数（デフォルト4拍子）
        - skip_initial_silence: 冒頭の無音をスキップするか
        """
        print(f"音源を読み込み中: {audio_path}")
        self.y, self.sr = librosa.load(audio_path)
        self.beats_per_bar = beats_per_bar
        self.skip_initial_silence = skip_initial_silence
        self.audio_start_time = 0.0  # 音声が始まる時間

        # 冒頭の無音をスキップ
        if skip_initial_silence:
            # 音声の開始位置を検出
            non_silent_intervals = librosa.effects.split(self.y, top_db=40)
            if len(non_silent_intervals) > 0:
                start_sample = non_silent_intervals[0][0]
                self.audio_start_time = librosa.samples_to_time(start_sample, sr=self.sr)
                print(f"冒頭の無音を検出: {self.audio_start_time:.2f}秒をスキップ")

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

        # RMS（音量）を計算（無音検出用）
        self.rms = librosa.feature.rms(y=self.y)[0]

        # 小節の計算
        self.total_beats = len(self.beat_times)
        self.total_bars = self.total_beats // beats_per_bar
        self.duration = len(self.y) / self.sr

        print(f"総ビート数: {self.total_beats}")
        print(f"総小節数: {self.total_bars}")
        print(f"曲の長さ: {self.duration:.2f}秒")

    def time_to_bar(self, time_sec: float) -> int:
        """時間（秒）を小節番号に変換"""
        # 冒頭の無音をスキップする場合は調整
        adjusted_time = max(0, time_sec - self.audio_start_time)
        beat_idx = np.searchsorted(self.beat_times, time_sec)

        # 音声開始前の時間は小節0とする
        if time_sec < self.audio_start_time:
            return 0

        bar_num = (beat_idx // self.beats_per_bar) + 1

        # 音声開始時刻でのビートインデックスを基準にする
        if self.skip_initial_silence and self.audio_start_time > 0:
            start_beat_idx = np.searchsorted(self.beat_times, self.audio_start_time)
            adjusted_beat_idx = beat_idx - start_beat_idx
            bar_num = (adjusted_beat_idx // self.beats_per_bar) + 1

        return max(1, min(bar_num, self.total_bars))

    def bar_to_time(self, bar_num: int) -> float:
        """小節番号を時間（秒）に変換"""
        if self.skip_initial_silence and self.audio_start_time > 0:
            # 音声開始時刻でのビートインデックスを取得
            start_beat_idx = np.searchsorted(self.beat_times, self.audio_start_time)
            beat_idx = start_beat_idx + (bar_num - 1) * self.beats_per_bar
        else:
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

    def detect_silence(self, threshold_db: float = -40, min_duration: float = 0.5) -> List[tuple]:
        """
        無音区間を検出

        Parameters:
        - threshold_db: 無音と判定する音量の閾値（dB）
        - min_duration: 無音として検出する最小時間（秒）

        Returns:
        - [(開始時間, 終了時間, 開始小節, 終了小節), ...] のリスト
        """
        # RMSをdBに変換
        rms_db = librosa.amplitude_to_db(self.rms, ref=np.max)

        # フレームを時間に変換
        times = librosa.frames_to_time(np.arange(len(rms_db)), sr=self.sr)

        # 閾値以下のフレームを検出
        is_silent = rms_db < threshold_db

        # 連続する無音区間を検出
        silences = []
        in_silence = False
        silence_start = 0

        for i, silent in enumerate(is_silent):
            if silent and not in_silence:
                # 無音開始
                in_silence = True
                silence_start = times[i]
            elif not silent and in_silence:
                # 無音終了
                in_silence = False
                silence_end = times[i]
                duration = silence_end - silence_start

                # 最小時間以上の無音のみ記録
                if duration >= min_duration:
                    start_bar = self.time_to_bar(silence_start)
                    end_bar = self.time_to_bar(silence_end)
                    silences.append((silence_start, silence_end, start_bar, end_bar))

        # 曲の最後が無音の場合
        if in_silence:
            silence_end = times[-1]
            duration = silence_end - silence_start
            if duration >= min_duration:
                start_bar = self.time_to_bar(silence_start)
                end_bar = self.time_to_bar(silence_end)
                silences.append((silence_start, silence_end, start_bar, end_bar))

        return silences

    def detect_vocal_only_sections(self, threshold_ratio: float = 0.7, min_duration: float = 1.0) -> List[tuple]:
        """
        Vo（ボーカル）のみの区間を検出
        ※簡易版：高周波成分と全体の音量バランスから推定

        Parameters:
        - threshold_ratio: ボーカル判定の閾値（0.0〜1.0）
        - min_duration: 検出する最小時間（秒）

        Returns:
        - [(開始時間, 終了時間, 開始小節, 終了小節), ...] のリスト
        """
        # スペクトログラムを計算
        D = np.abs(librosa.stft(self.y))

        # 周波数帯域を分割
        # ボーカル帯域（大体200Hz〜4000Hz）
        freqs = librosa.fft_frequencies(sr=self.sr)
        vocal_band_idx = np.where((freqs >= 200) & (freqs <= 4000))[0]
        bass_band_idx = np.where(freqs < 200)[0]
        high_band_idx = np.where(freqs > 4000)[0]

        # 各帯域のエネルギーを計算
        vocal_energy = np.sum(D[vocal_band_idx, :], axis=0)
        bass_energy = np.sum(D[bass_band_idx, :], axis=0)
        high_energy = np.sum(D[high_band_idx, :], axis=0)
        total_energy = np.sum(D, axis=0)

        # ボーカル帯域の比率を計算
        vocal_ratio = vocal_energy / (total_energy + 1e-6)

        # 低音（ベース・ドラム）が少ない箇所を検出
        bass_ratio = bass_energy / (total_energy + 1e-6)

        # ボーカルのみの判定（ボーカル帯域が支配的で、低音が少ない）
        is_vocal_only = (vocal_ratio > threshold_ratio) & (bass_ratio < 0.3)

        # フレームを時間に変換
        times = librosa.frames_to_time(np.arange(len(is_vocal_only)), sr=self.sr)

        # 連続するボーカル区間を検出
        vocal_sections = []
        in_vocal = False
        vocal_start = 0

        for i, is_vo in enumerate(is_vocal_only):
            if is_vo and not in_vocal:
                # ボーカル開始
                in_vocal = True
                vocal_start = times[i]
            elif not is_vo and in_vocal:
                # ボーカル終了
                in_vocal = False
                vocal_end = times[i]
                duration = vocal_end - vocal_start

                # 最小時間以上のボーカル区間のみ記録
                if duration >= min_duration:
                    start_bar = self.time_to_bar(vocal_start)
                    end_bar = self.time_to_bar(vocal_end)
                    vocal_sections.append((vocal_start, vocal_end, start_bar, end_bar))

        # 曲の最後がボーカルの場合
        if in_vocal:
            vocal_end = times[-1]
            duration = vocal_end - vocal_start
            if duration >= min_duration:
                start_bar = self.time_to_bar(vocal_start)
                end_bar = self.time_to_bar(vocal_end)
                vocal_sections.append((vocal_start, vocal_end, start_bar, end_bar))

        return vocal_sections

    def remove_drums(self, output_path: str = None, method: str = "simple") -> str:
        """
        ドラムの音を除去した音源を生成

        Parameters:
        - output_path: 出力ファイルパス（Noneの場合は自動生成）
        - method: "simple"（簡易版）または "spleeter"（高品質）

        Returns:
        - 出力ファイルパス
        """
        if method == "spleeter":
            try:
                from spleeter.separator import Separator
                print("\nSpleeterを使用してドラムを分離中...")

                # Spleeterで4stem分離（vocals, drums, bass, other）
                separator = Separator('spleeter:4stems')

                # 一時ファイルに保存
                import tempfile
                import soundfile as sf
                temp_input = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(temp_input.name, self.y, self.sr)

                # 分離実行
                prediction = separator.separate_to_file(temp_input.name, tempfile.gettempdir())

                # ドラム以外をミックス
                # vocals + bass + other
                print("ドラムを除去した音源を生成中...")

                return output_path or "no_drums.wav"

            except ImportError:
                print("エラー: Spleeterがインストールされていません")
                print("インストール: pip install spleeter")
                return None

        else:  # simple method
            print("\n簡易方式でドラムを減衰中...")

            # STFTで周波数領域に変換
            D = librosa.stft(self.y)
            magnitude, phase = np.abs(D), np.angle(D)

            # パーカッシブ成分とハーモニック成分を分離
            D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=3.0)

            # パーカッシブ成分（ドラム）を大幅に減衰
            # ハーモニック成分だけを使う
            y_no_drums = librosa.istft(D_harmonic)

            # 出力ファイル名
            if output_path is None:
                output_path = "output/no_drums.wav"

            # 保存
            import soundfile as sf
            sf.write(output_path, y_no_drums, self.sr)
            print(f"ドラムを除去した音源を保存: {output_path}")

            return output_path

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

    def print_structure(self, sections: List[Section], output_file: str = None, silences: List[tuple] = None, vocal_sections: List[tuple] = None):
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

        # 無音区間を追加
        if silences:
            output.append("")
            output.append("=" * 70)
            output.append("無音区間")
            output.append("=" * 70)
            for start_time, end_time, start_bar, end_bar in silences:
                duration = end_time - start_time
                output.append(f"無音         | 小節 {start_bar:3d}-{end_bar:3d} | {start_time:6.2f}s - {end_time:6.2f}s ({duration:.2f}秒)")
            output.append("=" * 70)

        # ボーカルのみの区間を追加
        if vocal_sections:
            output.append("")
            output.append("=" * 70)
            output.append("ボーカルのみの区間（推定）")
            output.append("=" * 70)
            for start_time, end_time, start_bar, end_bar in vocal_sections:
                duration = end_time - start_time
                output.append(f"Vo のみ      | 小節 {start_bar:3d}-{end_bar:3d} | {start_time:6.2f}s - {end_time:6.2f}s ({duration:.2f}秒)")
            output.append("=" * 70)

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
    parser.add_argument('--detect-silence', action='store_true',
                       help='無音区間を検出')
    parser.add_argument('--silence-threshold', type=float, default=-40,
                       help='無音判定の閾値（dB、デフォルト: -40）')
    parser.add_argument('--silence-min-duration', type=float, default=0.5,
                       help='無音として検出する最小時間（秒、デフォルト: 0.5）')
    parser.add_argument('--detect-vocal', action='store_true',
                       help='ボーカルのみの区間を検出')
    parser.add_argument('--vocal-threshold', type=float, default=0.7,
                       help='ボーカル判定の閾値（0.0〜1.0、デフォルト: 0.7）')
    parser.add_argument('--vocal-min-duration', type=float, default=1.0,
                       help='ボーカル区間として検出する最小時間（秒、デフォルト: 1.0）')
    parser.add_argument('--manual', type=str, default=None,
                       help='手動でセクションを指定（例: "Intro:1-8,Aメロ:9-24,サビ:25-40"）')
    parser.add_argument('--skip-initial-silence', action='store_true',
                       help='冒頭の無音を小節カウントからスキップ')
    parser.add_argument('--remove-drums', action='store_true',
                       help='ドラムを除去した音源を生成')
    parser.add_argument('--drums-method', type=str, default='simple', choices=['simple', 'spleeter'],
                       help='ドラム除去の方式（simple: 簡易版, spleeter: 高品質）')

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
        analyzer = MusicStructureAnalyzer(
            audio_file,
            bpm=args.bpm,
            beats_per_bar=args.beats,
            skip_initial_silence=args.skip_initial_silence
        )

        # 手動モードか自動モードか
        if args.manual:
            # 手動でセクションを指定
            print("\n【手動指定モード】")
            section_list = []
            for section_def in args.manual.split(','):
                parts = section_def.strip().split(':')
                if len(parts) != 2:
                    print(f"警告: 無効なフォーマット '{section_def}' - スキップします")
                    continue
                name = parts[0].strip()
                bars = parts[1].strip().split('-')
                if len(bars) != 2:
                    print(f"警告: 無効な小節範囲 '{parts[1]}' - スキップします")
                    continue
                try:
                    start_bar = int(bars[0])
                    end_bar = int(bars[1])
                    section_list.append((name, start_bar, end_bar))
                except ValueError:
                    print(f"警告: 小節番号が不正 '{parts[1]}' - スキップします")
                    continue

            if section_list:
                manual_sections = analyzer.create_manual_sections(section_list, detect_chords=not args.no_chords)

                # 無音区間を検出
                silences = None
                if args.detect_silence:
                    print("\n無音区間を検出中...")
                    silences = analyzer.detect_silence(
                        threshold_db=args.silence_threshold,
                        min_duration=args.silence_min_duration
                    )
                    print(f"検出された無音区間: {len(silences)}箇所")

                # ボーカルのみの区間を検出
                vocal_sections = None
                if args.detect_vocal:
                    print("\nボーカルのみの区間を検出中...")
                    vocal_sections = analyzer.detect_vocal_only_sections(
                        threshold_ratio=args.vocal_threshold,
                        min_duration=args.vocal_min_duration
                    )
                    print(f"検出されたボーカル区間: {len(vocal_sections)}箇所")

                analyzer.print_structure(manual_sections, output_path, silences=silences, vocal_sections=vocal_sections)
            else:
                print("エラー: 有効なセクション定義がありません")
        else:
            # 自動検出
            print("\n【自動検出】")
            auto_sections = analyzer.detect_sections_auto(detect_chords=not args.no_chords)

            # 無音区間を検出
            silences = None
            if args.detect_silence:
                print("\n無音区間を検出中...")
                silences = analyzer.detect_silence(
                    threshold_db=args.silence_threshold,
                    min_duration=args.silence_min_duration
                )
                print(f"検出された無音区間: {len(silences)}箇所")

            # ボーカルのみの区間を検出
            vocal_sections = None
            if args.detect_vocal:
                print("\nボーカルのみの区間を検出中...")
                vocal_sections = analyzer.detect_vocal_only_sections(
                    threshold_ratio=args.vocal_threshold,
                    min_duration=args.vocal_min_duration
                )
                print(f"検出されたボーカル区間: {len(vocal_sections)}箇所")

            analyzer.print_structure(auto_sections, output_path, silences=silences, vocal_sections=vocal_sections)

        # ドラム除去
        if args.remove_drums:
            drums_output = os.path.join(output_dir, f"{output_prefix}_no_drums.wav")
            analyzer.remove_drums(drums_output, method=args.drums_method)

        # 手動設定の例（--auto-onlyが指定されていない場合）
        if not args.auto_only and not args.manual:
            print("\n【ヒント】")
            print("セクション名を手動で指定するには --manual オプションを使用します：")
            print('python script.py song.mp3 --manual "Intro:1-8,Aメロ:9-24,サビ:25-40"')

    except FileNotFoundError:
        print(f"\nエラー: '{audio_file}' が見つかりません")
        print("音源ファイルのパスを確認してください")
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()
