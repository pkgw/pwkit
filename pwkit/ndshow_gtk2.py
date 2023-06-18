# -*- mode: python; coding: utf-8 -*-
# Copyright 2012-2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.ndshow_gtk2 - visualize data arrays with the Gtk+2 toolkit.

Functions:

view     - Show a GUI visualizing a 2D array.
cycle    - Show a GUI cycling through planes of a 3D array.

Classes:

Viewport - A GtkDrawingArea that renders a portion of an array.
Viewer   - A GUI window for visualizing a 2D array.
Cycler   - A GUI window for cycling through planes of a 3D array.


UI features of the viewport:

- click-drag to pan
- scrollwheel to zoom in/out (Ctrl to do so more aggressively)
-   (Shift to change color scale adjustment sensitivity)
- double-click to recenter
- shift-click-drag to adjust color scale (prototype)

Added by the toplevel window viewer:

- Ctrl-A to autoscale data to fit window
- Ctrl-E to center the data in the window
- Ctrl-F to fullscreen the window
- Escape to un-fullscreen it
- Ctrl-W to close the window
- Ctrl-1 to set scale to unity
- Ctrl-S to save the data to "data.png" under the current rendering options
  (but not zoomed to the current view of the data).

Added by cycler:

- Ctrl-K to move to next plane
- Ctrl-J to move to previous plane
- Ctrl-C to toggle automatic cycling

"""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str("Cycler Viewer Viewport cycle view").split()

import cairo, glib, gtk, numpy as np, sys, time

from . import PKError
from .data_gui_helpers import Clipper, ColorMapper


DRAG_TYPE_NONE = 0
DRAG_TYPE_PAN = 1
DRAG_TYPE_TUNER = 2

DEFAULT_WIN_WIDTH = 800
DEFAULT_WIN_HEIGHT = 600


class Viewport(gtk.DrawingArea):
    bgpattern = None

    getshape = None
    settuning = None
    getsurface = None
    onmotion = None
    drawoverlay = None

    centerx = 0
    centery = 0
    # The data pixel coordinate of the central pixel of the displayed window.

    scale = None
    # From data space to viewer space: e.g., scale = 2 means that each data
    # pixel occupies 2 pixels on-screen.

    needtune = True
    tunerx = 0
    tunery = 1.0
    tunerscale = 200

    drag_type = DRAG_TYPE_NONE
    drag_win_x0 = drag_win_y0 = None
    drag_dc_x0 = drag_dc_y0 = None

    def __init__(self):
        super(Viewport, self).__init__()
        self.add_events(
            gtk.gdk.POINTER_MOTION_MASK
            | gtk.gdk.BUTTON_PRESS_MASK
            | gtk.gdk.BUTTON_RELEASE_MASK
            | gtk.gdk.SCROLL_MASK
        )
        self.connect("expose-event", self._on_expose)
        self.connect("scroll-event", self._on_scroll)
        self.connect("button-press-event", self._on_button_press)
        self.connect("button-release-event", self._on_button_release)
        self.connect("motion-notify-event", self._on_motion_notify)

        self.bgpattern = cairo.SolidPattern(0.1, 0.1, 0.1)

    def set_shape_getter(self, getshape):
        if getshape is not None and not callable(getshape):
            raise ValueError()
        self.getshape = getshape
        return self

    def set_tuning_setter(self, settuning):
        if settuning is not None and not callable(settuning):
            raise ValueError()
        self.settuning = settuning
        self.needtune = True
        return self

    def set_surface_getter(self, getsurface):
        if getsurface is not None and not callable(getsurface):
            raise ValueError()
        self.getsurface = getsurface
        return self

    def set_motion_handler(self, onmotion):
        if onmotion is not None and not callable(onmotion):
            raise ValueError()
        self.onmotion = onmotion
        return self

    def set_overlay_drawer(self, drawoverlay):
        if drawoverlay is not None and not callable(drawoverlay):
            raise ValueError()
        self.drawoverlay = drawoverlay
        return self

    def autoscale(self):
        if self.allocation is None:
            raise PKError("must be called after allocation")
        if self.getshape is None:
            raise PKError("must be called after setting shape-getter")

        aw = self.allocation.width
        ah = self.allocation.height

        dw, dh = self.getshape()

        wratio = float(aw) / dw
        hratio = float(ah) / dh

        self.scale = min(wratio, hratio)
        self.centerx = 0.5 * (dw - 1)
        self.centery = 0.5 * (dh - 1)
        self.queue_draw()
        return self

    def center(self):
        if self.getshape is None:
            raise PKError("must be called after setting shape-getter")

        dw, dh = self.getshape()
        self.centerx = 0.5 * (dw - 1)
        self.centery = 0.5 * (dh - 1)
        self.queue_draw()
        return self

    def write_data_as_png(self, filename):
        if self.getshape is None:
            raise PKError("must be called after setting shape-getter")
        if self.getsurface is None:
            raise PKError("must be called after setting surface-getter")

        if self.needtune:
            self.settuning(self.tunerx, self.tunery)
            self.needtune = False

        dw, dh = self.getshape()
        surface, xoffset, yoffset = self.getsurface(0, 0, dw, dh)
        surface.write_to_png(filename)

    def write_view_as_png(self, filename):
        if self.getshape is None:
            raise PKError("must be called after setting shape-getter")
        if self.getsurface is None:
            raise PKError("must be called after setting surface-getter")
        if self.allocation is None:
            raise PKError("must be called after allocation")

        width = self.allocation.width
        height = self.allocation.height

        stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, width)
        assert stride % 4 == 0  # stride is in bytes
        viewdata = np.empty((height, stride // 4), dtype=np.uint32)
        viewsurface = cairo.ImageSurface.create_for_data(
            viewdata, cairo.FORMAT_ARGB32, width, height, stride
        )
        ctxt = cairo.Context(viewsurface)
        self._draw_in_context(ctxt, width, height)
        viewsurface.write_to_png(filename)

    def get_pointer_data_coords(self):
        if self.allocation is None:
            raise PKError("must be called after allocation")
        if self.scale is None:
            self.autoscale()

        x, y = self.get_pointer()
        dx = x - 0.5 * self.allocation.width
        dy = y - 0.5 * self.allocation.height
        datax = self.centerx + dx / self.scale
        datay = self.centery + dy / self.scale
        return datax, datay

    def _draw_in_context(self, ctxt, width, height):
        if self.getshape is None or self.getsurface is None:
            raise PKError(
                "must be called after setting shape-getter " "and surface-getter"
            )

        if self.scale is None:
            self.autoscale()
        if self.needtune:
            self.settuning(self.tunerx, self.tunery)
            self.needtune = False

        # Our data coordinates have integral pixel values being in the
        # centers of pixels; Cairo uses edges. That's the origin of the
        # minus one.
        seendatawidth = width / self.scale
        xoffset = 0.5 * (seendatawidth - 1) - self.centerx
        seendataheight = height / self.scale
        yoffset = 0.5 * (seendataheight - 1) - self.centery

        surface, xoffset, yoffset = self.getsurface(
            xoffset, yoffset, seendatawidth, seendataheight
        )

        ctxt.save()
        ctxt.set_source(self.bgpattern)
        ctxt.paint()
        ctxt.scale(self.scale, self.scale)
        ctxt.set_source_surface(surface, xoffset, yoffset)
        pat = ctxt.get_source()
        pat.set_extend(cairo.EXTEND_NONE)
        pat.set_filter(cairo.FILTER_NEAREST)
        ctxt.paint()
        ctxt.restore()

        if self.drawoverlay is not None:
            self.drawoverlay(ctxt, width, height, -xoffset, -yoffset, self.scale)

    def _on_expose(self, alsoself, event):
        if self.getshape is None or self.getsurface is None:
            return False

        self._draw_in_context(
            self.window.cairo_create(), self.allocation.width, self.allocation.height
        )
        return True

    def _on_scroll(self, alsoself, event):
        modmask = gtk.accelerator_get_default_mod_mask()

        if (event.state & modmask) in (0, gtk.gdk.CONTROL_MASK):
            oldscale = self.scale
            newscale = self.scale

            if event.state & modmask == gtk.gdk.CONTROL_MASK:
                factor = 1.2
            else:
                factor = 1.05

            if event.direction == gtk.gdk.SCROLL_UP:
                newscale *= factor

            if event.direction == gtk.gdk.SCROLL_DOWN:
                newscale /= factor

            if newscale == oldscale:
                return False

            self.scale = newscale
            self.queue_draw()
            return True

        if (event.state & modmask) == gtk.gdk.SHIFT_MASK:
            oldscale = self.tunerscale
            newscale = self.tunerscale

            if event.direction == gtk.gdk.SCROLL_UP:
                newscale *= 1.05

            if event.direction == gtk.gdk.SCROLL_DOWN:
                newscale /= 1.05

            if newscale == oldscale:
                return False

            self.tunerscale = newscale
            return True

        return False

    def _on_button_press(self, alsoself, event):
        modmask = gtk.accelerator_get_default_mod_mask()

        if event.type == gtk.gdk.BUTTON_PRESS and event.button == 1:
            self.grab_add()
            self.drag_win_x0 = event.x
            self.drag_win_y0 = event.y

            if (event.state & modmask) == 0:
                self.drag_type = DRAG_TYPE_PAN
                self.drag_dc_x0 = self.centerx
                self.drag_dc_y0 = self.centery
                return True

            if (event.state & modmask) == gtk.gdk.SHIFT_MASK:
                self.drag_type = DRAG_TYPE_TUNER
                self.drag_dc_x0 = self.tunerx
                self.drag_dc_y0 = self.tunery
                return True

            return False

        if event.type == gtk.gdk._2BUTTON_PRESS and event.button == 1:
            dx = event.x - 0.5 * self.allocation.width
            dy = event.y - 0.5 * self.allocation.height

            if (event.state & modmask) == 0:
                self.centerx += dx / self.scale
                self.centery += dy / self.scale
            elif (event.state & modmask) == gtk.gdk.SHIFT_MASK:
                self.tunerx += dx / self.tunerscale
                self.tunery += dy / self.tunerscale
                self.needtune = True
            else:
                return False

            self.queue_draw()
            # Prevent the drag-release code from running. (Double-click events
            # are preceded by single-click events.)
            self.grab_remove()
            self.drag_type = DRAG_TYPE_NONE
            return True

        return False

    def _on_button_release(self, alsoself, event):
        if event.type == gtk.gdk.BUTTON_RELEASE and event.button == 1:
            if self.drag_type == DRAG_TYPE_NONE:
                return False

            self.grab_remove()
            dx = self.drag_win_x0 - event.x
            dy = self.drag_win_y0 - event.y

            if self.drag_type == DRAG_TYPE_PAN:
                self.centerx = self.drag_dc_x0 + dx / self.scale
                self.centery = self.drag_dc_y0 + dy / self.scale
            elif self.drag_type == DRAG_TYPE_TUNER:
                self.tunerx = self.drag_dc_x0 - dx / self.tunerscale
                self.tunery = self.drag_dc_y0 - dy / self.tunerscale
                self.needtune = True
            else:
                return False

            self.drag_win_x0 = self.drag_win_y0 = None
            self.drag_dc_x0 = self.drag_dc_y0 = None
            self.drag_type = DRAG_TYPE_NONE
            self.queue_draw()
            return True

        return False

    def _on_motion_notify(self, alsoself, event):
        if self.onmotion is not None:
            dx = event.x - 0.5 * self.allocation.width
            dy = event.y - 0.5 * self.allocation.height
            datax = self.centerx + dx / self.scale
            datay = self.centery + dy / self.scale
            self.onmotion(datax, datay)

        if self.drag_type == DRAG_TYPE_NONE:
            return False
        elif self.drag_type == DRAG_TYPE_PAN:
            self.centerx = self.drag_dc_x0 + (self.drag_win_x0 - event.x) / self.scale
            self.centery = self.drag_dc_y0 + (self.drag_win_y0 - event.y) / self.scale
        elif self.drag_type == DRAG_TYPE_TUNER:
            self.tunerx = (
                self.drag_dc_x0 + (event.x - self.drag_win_x0) / self.tunerscale
            )
            self.tunery = (
                self.drag_dc_y0 + (event.y - self.drag_win_y0) / self.tunerscale
            )
            self.needtune = True

        self.queue_draw()
        return True


class Viewer(object):
    def __init__(
        self,
        title="Array Viewer",
        default_width=DEFAULT_WIN_WIDTH,
        default_height=DEFAULT_WIN_HEIGHT,
    ):
        self.viewport = Viewport()
        self.win = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.win.set_title(title)
        self.win.set_default_size(default_width, default_height)
        self.win.connect("key-press-event", self._on_key_press)

        vb = gtk.VBox()
        vb.pack_start(self.viewport, True, True, 2)
        hb = gtk.HBox()
        vb.pack_start(hb, False, True, 2)

        self.status_label = gtk.Label()
        self.status_label.set_alignment(0, 0.5)
        hb.pack_start(self.status_label, True, True, 2)

        self.status_label.set_markup("Temp")

        self.win.add(vb)

    def set_shape_getter(self, getshape):
        self.viewport.set_shape_getter(getshape)
        return self

    def set_tuning_setter(self, settuning):
        self.viewport.set_tuning_setter(settuning)
        return self

    def set_surface_getter(self, getsurface):
        self.viewport.set_surface_getter(getsurface)
        return self

    def set_status_formatter(self, fmtstatus):
        def onmotion(x, y):
            self.status_label.set_markup(fmtstatus(x, y))

        self.viewport.set_motion_handler(onmotion)
        return self

    def set_overlay_drawer(self, drawoverlay):
        self.viewport.set_overlay_drawer(drawoverlay)
        return self

    def _on_key_press(self, widget, event):
        kn = gtk.gdk.keyval_name(event.keyval)
        modmask = gtk.accelerator_get_default_mod_mask()
        isctrl = (event.state & modmask) == gtk.gdk.CONTROL_MASK

        if kn == "a" and isctrl:
            self.viewport.autoscale()
            return True

        if kn == "e" and isctrl:
            self.viewport.center()
            return True

        if kn == "f" and isctrl:
            self.win.fullscreen()
            return True

        if kn == "Escape":
            self.win.unfullscreen()
            return True

        if kn == "w" and isctrl:
            self.win.destroy()
            return True

        if kn == "1" and isctrl:
            self.viewport.scale = 1.0
            self.viewport.queue_draw()
            return True

        if kn == "s" and isctrl:
            print("Writing data.png ...", end="")
            sys.stdout.flush()
            self.viewport.write_data_as_png("data.png")
            print("done")
            return True

        return False


def view(
    array,
    title="Array Viewer",
    colormap="black_to_blue",
    toworld=None,
    drawoverlay=None,
    yflip=False,
):
    clipper = Clipper()
    clipper.alloc_buffer(array)
    clipper.set_tile_size()
    clipper.default_bounds(array)
    processed = clipper.buffer

    mapper = ColorMapper(colormap)
    mapper.alloc_buffer(array)
    mapper.set_tile_size()

    h, w = array.shape
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    assert stride % 4 == 0  # stride is in bytes
    assert stride == 4 * w  # size of buffer is set in mapper
    imagesurface = cairo.ImageSurface.create_for_data(
        mapper.buffer, cairo.FORMAT_ARGB32, w, h, stride
    )

    def getshape():
        return w, h

    orig_min = clipper.dmin
    orig_span = clipper.dmax - orig_min

    def settuning(tunerx, tunery):
        clipper.dmin = orig_span * tunerx + orig_min
        clipper.dmax = orig_span * tunery + orig_min
        clipper.invalidate()
        mapper.invalidate()

    def getsurface(xoffset, yoffset, width, height):
        pxofs = max(int(np.floor(-xoffset)), 0)
        pyofs = max(int(np.floor(-yoffset)), 0)
        pw = min(int(np.ceil(width)), w - pxofs)
        ph = min(int(np.ceil(height)), h - pyofs)

        clipper.ensure_region_updated(array, pxofs, pyofs, pw, ph)
        mapper.ensure_region_updated(processed, pxofs, pyofs, pw, ph)

        return imagesurface, xoffset, yoffset

    # I originally had the is_masked call inside fmtstatus and somehow it
    # ended up causing large lags in the label updates. Can't be that
    # CPU-intensive, right??

    nomask = not np.ma.is_masked(array) or array.mask is np.ma.nomask

    if toworld is None:

        def fmtstatus(x, y):
            s = ""
            row = int(np.floor(y + 0.5))
            col = int(np.floor(x + 0.5))
            if row >= 0 and col >= 0 and row < h and col < w:
                if nomask or not array.mask[row, col]:
                    s += "%g " % array[row, col]
            if yflip:
                y = h - 1 - y
                row = h - 1 - row
            return s + "[%d,%d] x=%.1f y=%.1f" % (row, col, x, y)

    else:
        from .astutil import fmthours, fmtdeglat

        def fmtstatus(x, y):
            s = ""
            row = int(np.floor(y + 0.5))
            col = int(np.floor(x + 0.5))
            if row >= 0 and col >= 0 and row < h and col < w:
                if nomask or not array.mask[row, col]:
                    s += "%g " % array[row, col]
            if yflip:
                y = h - 1 - y
                row = h - 1 - row
            lat, lon = toworld([y, x])
            s += "[%d,%d] x=%.1f y=%.1f lat=%s lon=%s" % (
                row,
                col,
                x,
                y,
                fmtdeglat(lat),
                fmthours(lon),
            )
            return s

    viewer = Viewer(title=title)
    viewer.set_shape_getter(getshape)
    viewer.set_tuning_setter(settuning)
    viewer.set_surface_getter(getsurface)
    viewer.set_status_formatter(fmtstatus)
    viewer.set_overlay_drawer(drawoverlay)
    viewer.win.show_all()
    viewer.win.connect("destroy", gtk.main_quit)
    gtk.main()


class Cycler(Viewer):
    getn = None
    getshapei = None
    getdesci = None
    settuningi = None
    getsurfacei = None

    i = None
    sourceid = None
    needtune = None

    def __init__(
        self,
        title="Array Cycler",
        default_width=DEFAULT_WIN_WIDTH,
        default_height=DEFAULT_WIN_HEIGHT,
        cadence=0.6,
    ):
        self.cadence = cadence

        self.viewport = Viewport()
        self.win = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.win.set_title(title)
        self.win.set_default_size(default_width, default_height)
        self.win.connect("key-press-event", self._on_key_press)
        self.win.connect("realize", self._on_realize)
        self.win.connect("unrealize", self._on_unrealize)

        vb = gtk.VBox()
        vb.pack_start(self.viewport, True, True, 2)
        hb = gtk.HBox()
        vb.pack_start(hb, False, True, 2)
        self.status_label = gtk.Label()
        self.status_label.set_alignment(0, 0.5)
        hb.pack_start(self.status_label, True, True, 2)
        self.plane_label = gtk.Label()
        self.plane_label.set_alignment(0, 0.5)
        hb.pack_start(self.plane_label, True, True, 2)
        self.desc_label = gtk.Label()
        hb.pack_start(self.desc_label, True, True, 2)
        self.cycle_tbutton = gtk.ToggleButton("Cycle")
        hb.pack_start(self.cycle_tbutton, False, True, 2)
        self.win.add(vb)

        self.viewport.set_shape_getter(self._get_shape)
        self.viewport.set_surface_getter(self._get_surface)
        self.viewport.set_tuning_setter(self._set_tuning)

        self.cycle_tbutton.set_active(True)

    def set_n_getter(self, getn):
        if not callable(getn):
            raise ValueError("not callable")
        self.getn = getn
        return self

    def _get_shape(self):
        if self.i is None:
            self.set_current(0)
        return self.getshapei(self.i)

    def set_shape_getter(self, getshapei):
        if not callable(getshapei):
            raise ValueError("not callable")
        self.getshapei = getshapei
        return self

    def set_desc_getter(self, getdesci):
        if not callable(getdesci):
            raise ValueError("not callable")
        self.getdesci = getdesci
        return self

    def _set_tuning(self, tunerx, tunery):
        if self.i is None:
            self.set_current(0)
        self.settuningi(self.i, tunerx, tunery)
        self.needtune.fill(True)
        self.needtune[self.i] = False

    def set_tuning_setter(self, settuningi):
        if not callable(settuningi):
            raise ValueError("not callable")
        self.settuningi = settuningi
        self.viewport.set_tuning_setter(self._set_tuning)  # force retune
        return self

    def _get_surface(self, xoffset, yoffset, width, height):
        if self.i is None:
            self.set_current(0)
        return self.getsurfacei(self.i, xoffset, yoffset, width, height)

    def set_surface_getter(self, getsurfacei):
        if not callable(getsurfacei):
            raise ValueError("not callable")
        self.getsurfacei = getsurfacei
        return self

    def set_status_formatter(self, fmtstatusi):
        if fmtstatusi is None:
            self.viewport.set_motion_handler(None)
        else:

            def onmotion(x, y):
                self.status_label.set_markup(fmtstatusi(self.i, x, y))

            self.viewport.set_motion_handler(onmotion)
        return self

    def set_overlay_drawer(self, drawoverlay):
        self.viewport.set_overlay_drawer(drawoverlay)
        return self

    def set_current(self, index):
        n = self.getn()
        index = index % n

        if self.needtune is None or self.needtune.size != n:
            self.needtune = np.ones(n, dtype=bool)

        if index == self.i:
            return

        if self.needtune[index]:
            # Force the viewport to call settuning the next time it needs to
            self.viewport.set_tuning_setter(self._set_tuning)

        self.i = index
        self.plane_label.set_markup("<b>Current plane:</b> %d of %d" % (self.i + 1, n))
        self.desc_label.set_text(self.getdesci(self.i))

        if self.viewport.onmotion is not None:
            datax, datay = self.viewport.get_pointer_data_coords()
            self.viewport.onmotion(datax, datay)

        self.viewport.queue_draw()

    def _on_realize(self, widget):
        if self.sourceid is not None:
            return
        self.sourceid = glib.timeout_add(int(self.cadence * 1000), self._do_cycle)

    def _on_unrealize(self, widget):
        if self.sourceid is None:
            return
        glib.source_remove(self.sourceid)
        self.sourceid = None

    def _do_cycle(self):
        if self.cycle_tbutton.get_active():
            self.set_current(self.i + 1)
        return True

    def _on_key_press(self, widget, event):
        kn = gtk.gdk.keyval_name(event.keyval)
        modmask = gtk.accelerator_get_default_mod_mask()
        isctrl = (event.state & modmask) == gtk.gdk.CONTROL_MASK

        if kn == "j" and isctrl:
            self.set_current(self.i - 1)
            return True

        if kn == "k" and isctrl:
            self.set_current(self.i + 1)
            return True

        if kn == "c" and isctrl:
            self.cycle_tbutton.set_active(not self.cycle_tbutton.get_active())
            return True

        return super(Cycler, self)._on_key_press(widget, event)


def cycle(
    arrays, descs=None, cadence=0.6, toworlds=None, drawoverlay=None, yflip=False
):
    n = len(arrays)
    amin = amax = h = w = None

    if descs is None:
        descs = [""] * n

    for array in arrays:
        thish, thisw = array.shape
        thismin, thismax = array.min(), array.max()

        if not np.isfinite(thismin):
            thismin = array[np.ma.where(np.isfinite(array))].min()
        if not np.isfinite(thismax):
            thismax = array[np.ma.where(np.isfinite(array))].max()

        if amin is None:
            w, h, amin, amax = thisw, thish, thismin, thismax
        else:
            if thisw != w:
                raise ValueError("array widths not all equal")
            if thish != h:
                raise ValueError("array heights not all equal")

            amin = min(amin, thismin)
            amax = max(amax, thismax)

    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    assert stride % 4 == 0  # stride is in bytes
    imgdata = np.empty((n, h, stride // 4), dtype=np.uint32)
    fixed = np.empty((n, h, w), dtype=np.int32)
    antimask = np.empty((n, h, w), dtype=bool)
    surfaces = [None] * n

    imgdata.fill(0xFF000000)

    for i, array in enumerate(arrays):
        surfaces[i] = cairo.ImageSurface.create_for_data(
            imgdata[i], cairo.FORMAT_ARGB32, w, h, stride
        )

        if np.ma.is_masked(array):
            filled = array.filled(amin)
            antimask[i] = ~array.mask
        else:
            filled = array
            antimask[i].fill(True)

        fixed[i] = (filled - amin) * (0x0FFFFFF0 / (amax - amin))

    def getn():
        return n

    def getshapei(i):
        return w, h

    def getdesci(i):
        return descs[i]

    clipped = np.zeros((h, w), dtype=np.int32)  # scratch array

    def settuningi(i, tunerx, tunery):
        np.bitwise_and(imgdata[i], 0xFF000000, imgdata[i])

        fmin = int(0x0FFFFFF0 * tunerx)
        fmax = int(0x0FFFFFF0 * tunery)

        if fmin == fmax:
            np.add(imgdata[i], 255 * (fixed[i] > fmin), imgdata[i])
        else:
            np.clip(fixed[i], fmin, fmax, clipped)
            np.subtract(clipped, fmin, clipped)
            np.multiply(clipped, 255.0 / (fmax - fmin), clipped)
            np.add(imgdata[i], clipped, imgdata[i])

        np.multiply(imgdata[i], antimask[i], imgdata[i])

    def getsurfacei(i, xoffset, yoffset, width, height):
        return surfaces[i], xoffset, yoffset

    # see comment in view()
    nomasks = [not np.ma.is_masked(a) or a.mask is np.ma.nomask for a in arrays]

    if toworlds is None:
        toworlds = [None] * n

    from .astutil import fmthours, fmtdeglat

    def fmtstatusi(i, x, y):
        s = ""
        row = int(np.floor(y + 0.5))
        col = int(np.floor(x + 0.5))
        if row >= 0 and col >= 0 and row < h and col < w:
            if nomasks[i] or not arrays[i].mask[row, col]:
                s += "%g " % arrays[i][row, col]
        if yflip:
            y = h - 1 - y
            row = h - 1 - row
        s += "[%d,%d] x=%.1f y=%.1f" % (row, col, x, y)
        if toworlds[i] is not None:
            lat, lon = toworlds[i]([y, x])
            s += " lat=%s lon=%s" % (fmtdeglat(lat), fmthours(lon))
        return s

    cycler = Cycler()
    cycler.set_n_getter(getn)
    cycler.set_shape_getter(getshapei)
    cycler.set_desc_getter(getdesci)
    cycler.set_tuning_setter(settuningi)
    cycler.set_surface_getter(getsurfacei)
    cycler.set_status_formatter(fmtstatusi)
    cycler.set_overlay_drawer(drawoverlay)
    cycler.win.show_all()
    cycler.win.connect("destroy", gtk.main_quit)

    gtk.main()
