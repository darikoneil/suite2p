"""Trace plotting controls for the suite2p GUI."""

import numpy as np
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QLineEdit, QPushButton
from .trace_utils import (
    build_trace_time_axis,
    convert_behavior_time_axis,
    gaussian_smooth_trace,
)


def plot_trace(parent) -> None:
    """Plot the currently selected ROI traces in the main GUI."""
    parent.p3.clear()
    ax = parent.p3.getAxis("left")
    trace_axis = _set_x_axis(parent)
    if len(parent.imerge) == 1:
        n = parent.imerge[0]
        f = _transform_trace(parent.Fcell[n, :], parent)
        fneu = _transform_trace(parent.Fneu[n, :], parent)
        sp = _transform_trace(parent.Spks[n, :], parent)
        if np.ptp(fneu) == 0:
            fmax = f.max()
            fmin = f.min()
        else:
            fmax = np.maximum(f.max(), fneu.max())
            fmin = np.minimum(f.min(), fneu.min())
        if sp.max() > 0:
            sp = sp / sp.max()
        sp *= fmax - fmin
        if parent.tracesOn:
            parent.p3.plot(trace_axis, f, pen="c")
        if parent.neuropilOn:
            parent.p3.plot(trace_axis, fneu, pen="r")
        if parent.deconvOn:
            parent.p3.plot(trace_axis, sp + fmin, pen=(255, 255, 255, 150))
        parent.fmin = fmin
        parent.fmax = fmax
        ax.setTicks(None)
    else:
        nmax = int(parent.ncedit.text())
        kspace = 1.0 / parent.sc
        ttick = []
        pmerge = parent.imerge[: np.minimum(len(parent.imerge), nmax)]
        k = len(pmerge) - 1
        activity_mode = parent.activityMode
        favg = np.zeros((parent.Fcell.shape[1],))
        for n in pmerge[::-1]:
            if activity_mode == 0:
                f = parent.Fcell[n, :]
            elif activity_mode == 1:
                f = parent.Fneu[n, :]
            elif activity_mode == 2:
                f = parent.Fcell[n, :] - 0.7 * parent.Fneu[n, :]
            else:
                f = parent.Spks[n, :]
            f = _transform_trace(f, parent)
            favg += f.flatten()
            fmax = f.max()
            fmin = f.min()
            if fmax > fmin:
                f = (f - fmin) / (fmax - fmin)
            else:
                f = np.zeros_like(f)
            rgb = parent.colors["cols"][0][n, :]
            parent.p3.plot(trace_axis, f + k * kspace, pen=rgb)
            ttick.append((k * kspace + f.mean(), str(n)))
            k -= 1
        bsc = len(pmerge) / 25 + 1
        favg -= favg.min()
        if favg.max() > 0:
            favg /= favg.max()
        parent.fmin = 0
        if len(pmerge) > 5:
            parent.p3.plot(trace_axis, -1 * bsc + favg * bsc, pen=(140, 140, 140))
            parent.fmin = -1 * bsc
        if parent.bloaded:
            behavior_axis = convert_behavior_time_axis(
                parent.beh_time,
                trace_axis,
                parent.timeAxisOn,
                _current_frame_rate(parent),
            )
            parent.p3.plot(trace_axis, -1 * bsc + favg * bsc, pen=(140, 140, 140))
            parent.p3.plot(behavior_axis, -1 * bsc + parent.beh * bsc, pen="w")
            parent.fmin = -1 * bsc
        parent.fmax = (len(pmerge) - 1) * kspace + 1
        ax.setTicks([ttick])
    parent.p3.setXRange(parent.trace_xmin, parent.trace_xmax, padding=0.0)
    parent.p3.setYRange(parent.fmin, parent.fmax)


def make_buttons(parent, b0: int) -> int:
    """Create the trace-panel controls for the main GUI.

    Args:
        parent: Parent GUI window.
        b0: Starting row index for the controls.

    Returns:
        The last row index used by the trace controls.
    """
    qlabel = QLabel(parent)
    qlabel.setText("Activity mode")
    parent.l0.addWidget(qlabel, b0, 0, 1, 1)
    parent.comboBox = QComboBox(parent)
    parent.comboBox.setFixedWidth(100)
    parent.l0.addWidget(parent.comboBox, b0 + 1, 0, 1, 1)
    parent.comboBox.addItem("F")
    parent.comboBox.addItem("Fneu")
    parent.comboBox.addItem("F - 0.7*Fneu")
    parent.comboBox.addItem("deconvolved")
    parent.activityMode = 3
    parent.comboBox.setCurrentIndex(parent.activityMode)
    parent.comboBox.currentIndexChanged.connect(parent.mode_change)

    parent.level = 1
    parent.arrowButtons = [
        QPushButton(" \u25b2"),
        QPushButton(" \u25bc"),
    ]
    parent.arrowButtons[0].clicked.connect(lambda: expand_trace(parent))
    parent.arrowButtons[1].clicked.connect(lambda: collapse_trace(parent))
    button_row = 0
    for btn in parent.arrowButtons:
        btn.setMaximumWidth(22)
        btn.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        btn.setStyleSheet(parent.styleUnpressed)
        parent.l0.addWidget(btn, b0 + button_row, 1, 1, 1, QtCore.Qt.AlignRight)
        button_row += 1

    parent.pmButtons = [QPushButton(" +"), QPushButton(" -")]
    parent.pmButtons[0].clicked.connect(lambda: expand_scale(parent))
    parent.pmButtons[1].clicked.connect(lambda: collapse_scale(parent))
    button_row = 0
    parent.sc = 2
    for btn in parent.pmButtons:
        btn.setMaximumWidth(22)
        btn.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        btn.setStyleSheet(parent.styleUnpressed)
        parent.l0.addWidget(btn, b0 + button_row, 1, 1, 1)
        button_row += 1
    parent.l0.addWidget(
        QLabel("<font color='white'>max # plotted:</font>"),
        b0 + 2,
        0,
        1,
        1,
    )
    b0 += 3
    parent.ncedit = QLineEdit(parent)
    parent.ncedit.setValidator(QtGui.QIntValidator(0, 400))
    parent.ncedit.setText("40")
    parent.ncedit.setFixedWidth(35)
    parent.ncedit.setAlignment(QtCore.Qt.AlignRight)
    parent.ncedit.returnPressed.connect(lambda: nc_chosen(parent))
    parent.l0.addWidget(parent.ncedit, b0, 0, 1, 1)

    parent.l0.setVerticalSpacing(4)
    parent.checkBoxd = QCheckBox("deconv [N]")
    parent.checkBoxd.setStyleSheet("color: white;")
    parent.checkBoxd.toggled.connect(lambda: deconv_on(parent))
    parent.deconvOn = True
    parent.checkBoxd.toggle()
    parent.l0.addWidget(parent.checkBoxd, b0, 3, 1, 2)

    parent.l0.setVerticalSpacing(4)
    parent.checkBoxn = QCheckBox("neuropil [B]")
    parent.checkBoxn.setStyleSheet("color: red;")
    parent.checkBoxn.toggled.connect(lambda: neuropil_on(parent))
    parent.neuropilOn = True
    parent.checkBoxn.toggle()
    parent.l0.addWidget(parent.checkBoxn, b0, 5, 1, 2)

    parent.l0.setVerticalSpacing(4)
    parent.checkBoxt = QCheckBox("raw fluor [V]")
    parent.checkBoxt.setStyleSheet("color: cyan;")
    parent.checkBoxt.toggled.connect(lambda: traces_on(parent))
    parent.tracesOn = True
    parent.checkBoxt.toggle()
    parent.l0.addWidget(parent.checkBoxt, b0, 7, 1, 2)

    b0 += 1
    parent.checkBoxSmooth = QCheckBox("gaussian smooth")
    parent.checkBoxSmooth.setStyleSheet("color: white;")
    parent.checkBoxSmooth.toggled.connect(lambda: smoothing_on(parent))
    parent.smoothTracesOn = False
    parent.l0.addWidget(parent.checkBoxSmooth, b0, 3, 1, 2)

    parent.l0.addWidget(
        QLabel("<font color='white'>sigma (frames):</font>"), b0, 5, 1, 2
    )
    parent.smoothingSigmaEdit = QLineEdit(parent)
    parent.smoothingSigmaEdit.setValidator(QtGui.QDoubleValidator(0.0, 1e6, 3))
    parent.smoothingSigmaEdit.setText("1.0")
    parent.smoothingSigmaEdit.setFixedWidth(50)
    parent.smoothingSigmaEdit.setAlignment(QtCore.Qt.AlignRight)
    parent.smoothingSigmaEdit.returnPressed.connect(
        lambda: trace_settings_changed(parent)
    )
    parent.l0.addWidget(parent.smoothingSigmaEdit, b0, 7, 1, 1)

    parent.checkBoxTime = QCheckBox("time axis")
    parent.checkBoxTime.setStyleSheet("color: white;")
    parent.checkBoxTime.toggled.connect(lambda: time_axis_on(parent))
    parent.timeAxisOn = False
    parent.l0.addWidget(parent.checkBoxTime, b0, 9, 1, 2)

    parent.l0.addWidget(QLabel("<font color='white'>fps:</font>"), b0, 11, 1, 1)
    parent.frameRateEdit = QLineEdit(parent)
    parent.frameRateEdit.setValidator(QtGui.QDoubleValidator(1e-9, 1e6, 6))
    parent.frameRateEdit.setPlaceholderText("auto")
    parent.frameRateEdit.setFixedWidth(55)
    parent.frameRateEdit.setAlignment(QtCore.Qt.AlignRight)
    parent.frameRateEdit.returnPressed.connect(lambda: trace_settings_changed(parent))
    parent.l0.addWidget(parent.frameRateEdit, b0, 12, 1, 1)
    return b0


def expand_scale(parent) -> None:
    """Increase the vertical spacing between stacked traces."""
    parent.sc += 0.5
    parent.sc = np.minimum(10, parent.sc)
    plot_trace(parent)
    parent.show()


def collapse_scale(parent) -> None:
    """Decrease the vertical spacing between stacked traces."""
    parent.sc -= 0.5
    parent.sc = np.maximum(0.5, parent.sc)
    plot_trace(parent)
    parent.show()


def expand_trace(parent) -> None:
    """Increase the relative height of the trace panel."""
    parent.level += 1
    parent.level = np.minimum(5, parent.level)
    parent.win.ci.layout.setRowStretchFactor(1, parent.level)


def collapse_trace(parent) -> None:
    """Decrease the relative height of the trace panel."""
    parent.level -= 1
    parent.level = np.maximum(1, parent.level)
    parent.win.ci.layout.setRowStretchFactor(1, parent.level)


def nc_chosen(parent) -> None:
    """Replot after changing the maximum number of displayed ROIs."""
    if parent.loaded:
        plot_trace(parent)
        parent.show()


def deconv_on(parent) -> None:
    """Toggle deconvolved-trace visibility."""
    if parent.loaded:
        parent.deconvOn = parent.checkBoxd.isChecked()
        plot_trace(parent)
        parent.win.show()
        parent.show()


def neuropil_on(parent) -> None:
    """Toggle neuropil-trace visibility."""
    if parent.loaded:
        parent.neuropilOn = parent.checkBoxn.isChecked()
        plot_trace(parent)
        parent.win.show()
        parent.show()


def traces_on(parent) -> None:
    """Toggle raw-fluorescence visibility."""
    if parent.loaded:
        parent.tracesOn = parent.checkBoxt.isChecked()
        plot_trace(parent)
        parent.win.show()
        parent.show()


def smoothing_on(parent) -> None:
    """Toggle Gaussian smoothing for displayed traces."""
    parent.smoothTracesOn = parent.checkBoxSmooth.isChecked()
    if parent.loaded:
        plot_trace(parent)
        parent.win.show()
        parent.show()


def time_axis_on(parent) -> None:
    """Toggle between frame units and seconds on the trace x-axis."""
    parent.timeAxisOn = parent.checkBoxTime.isChecked()
    if parent.loaded:
        plot_trace(parent)
        parent.win.show()
        parent.show()


def trace_settings_changed(parent) -> None:
    """Replot traces after editing smoothing or frame-rate fields."""
    if parent.loaded:
        plot_trace(parent)
        parent.win.show()
        parent.show()


def _current_frame_rate(parent) -> float:
    """Return the active frame rate requested in the GUI."""
    text = parent.frameRateEdit.text().strip()
    if not text:
        return float(parent.ops.get("fs", 1.0)) if hasattr(parent, "ops") else 1.0
    frame_rate = float(text)
    if frame_rate <= 0:
        raise ValueError("Frame rate must be positive.")
    return frame_rate


def _current_smoothing_sigma(parent) -> float:
    """Return the active Gaussian smoothing width requested in the GUI."""
    text = parent.smoothingSigmaEdit.text().strip()
    if not text:
        return 0.0
    sigma = float(text)
    if sigma < 0:
        raise ValueError("Gaussian smoothing sigma must be non-negative.")
    return sigma


def _transform_trace(trace: np.ndarray, parent) -> np.ndarray:
    """Apply the active display transform to one trace."""
    sigma = _current_smoothing_sigma(parent) if parent.smoothTracesOn else 0.0
    return gaussian_smooth_trace(trace, sigma)


def _set_x_axis(parent) -> np.ndarray:
    """Update the trace x-axis state and return the axis values."""
    trace_axis, axis_label = build_trace_time_axis(
        n_frames=parent.Fcell.shape[1],
        use_time_axis=parent.timeAxisOn,
        frame_rate=_current_frame_rate(parent),
    )
    parent.trace_axis = trace_axis
    parent.trace_xmin = float(trace_axis[0]) if trace_axis.size else 0.0
    parent.trace_xmax = float(trace_axis[-1]) if trace_axis.size else 0.0
    if parent.trace_xmax <= parent.trace_xmin:
        parent.trace_xmax = parent.trace_xmin + 1.0
    parent.p3.setLimits(xMin=parent.trace_xmin, xMax=parent.trace_xmax)
    parent.p3.setLabel("bottom", axis_label)
    return trace_axis
