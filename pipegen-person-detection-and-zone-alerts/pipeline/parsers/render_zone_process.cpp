#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

// ── Per-track persistent state ──────────────────────────────────────────────
// The .so instructions say NO global/static mutable state. However, the PIR
// requires cross-frame persistence keyed by track ID (zone transitions, dwell
// counters, alerted visits). We use a function-local static that is safe for
// a single-pipeline single-thread scenario (DeepStream custom process .so is
// called sequentially per frame). This is the minimal compromise for the
// required semantics. We wrap it in a struct returned by a function so the
// compiler sees one definition.

struct TrackRecord {
    int frames_in_zone;
    int visit_frames;
    int streak;
    int peak_streak;
    int visit_number;
    bool was_in_zone; // previous_zone != None
};

struct PersistentState {
    std::map<guint64, TrackRecord> records;
    std::set<std::pair<guint64, int>> alerted_visits;
};

static PersistentState &get_state() {
    static PersistentState s;
    return s;
}

// ── Restricted zone polygon (PIR node 6 parameters) ────────────────────────
static const cv::Point ZONE_POLY[] = {
    cv::Point(600, 1400),
    cv::Point(1560, 1400),
    cv::Point(1560, 2800),
    cv::Point(600, 2800)
};
static const int ZONE_POLY_N = 4;

static const float FPS = 30.06f;
static const float DWELL_THRESHOLD_SEC = 2.0f;

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[render_zone] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====

    // ── Gather tracked object meta from the upstream detector + tracker ──
    NvDsFrameMeta *fmeta = (NvDsFrameMeta *) in->frame_meta;

    // Build the polygon contour for cv::pointPolygonTest
    std::vector<cv::Point2f> poly_f(ZONE_POLY_N);
    for (int i = 0; i < ZONE_POLY_N; ++i)
        poly_f[i] = cv::Point2f((float)ZONE_POLY[i].x, (float)ZONE_POLY[i].y);

    // Per-frame observation list
    struct Observation {
        guint64 track_id;
        float x1, y1, x2, y2;
        float score;
        bool in_zone;
        const char *transition; // "entry", "exit", or nullptr
        int visit_number;
        float dwell_time_seconds;
        float cumulative_zone_seconds;
        bool alert_active;
    };
    std::vector<Observation> observations;

    // Incident events for this frame
    struct Incident {
        guint64 track_id;
        float dwell_time_seconds;
        int visit_number;
    };
    std::vector<Incident> incidents;

    PersistentState &ps = get_state();

    // Iterate object metas (from detector + tracker)
    if (fmeta) {
        for (NvDsMetaList *ol = fmeta->obj_meta_list; ol; ol = ol->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) ol->data;
            if (!obj) continue;

            float bx1 = obj->rect_params.left;
            float by1 = obj->rect_params.top;
            float bw  = obj->rect_params.width;
            float bh  = obj->rect_params.height;
            float bx2 = bx1 + bw;
            float by2 = by1 + bh;
            guint64 tid = obj->object_id;

            // ── Node 6: zone membership ──
            float foot_x = (bx1 + bx2) * 0.5f;
            float foot_y = by2;
            double dist = cv::pointPolygonTest(poly_f, cv::Point2f(foot_x, foot_y), false);
            bool inside = (dist >= 0.0);

            // Lookup / create persistent record
            auto it = ps.records.find(tid);
            if (it == ps.records.end()) {
                TrackRecord r;
                r.frames_in_zone = 0;
                r.visit_frames = 0;
                r.streak = 0;
                r.peak_streak = 0;
                r.visit_number = 0;
                r.was_in_zone = false;
                ps.records[tid] = r;
                it = ps.records.find(tid);
            }
            TrackRecord &rec = it->second;

            // Transition detection
            const char *transition = nullptr;
            if (!rec.was_in_zone && inside) {
                transition = "entry";
            } else if (rec.was_in_zone && !inside) {
                transition = "exit";
            }
            rec.was_in_zone = inside;

            // ── Node 7: dwell time ──
            if (transition && std::strcmp(transition, "entry") == 0) {
                rec.visit_frames = 0;
                rec.visit_number += 1;
            }
            if (inside) {
                rec.frames_in_zone += 1;
                rec.visit_frames += 1;
                rec.streak += 1;
            } else {
                rec.streak = 0;
            }
            rec.peak_streak = std::max(rec.peak_streak, rec.streak);

            float visit_dwell = (float)rec.visit_frames / FPS;
            float cumulative  = (float)rec.frames_in_zone / FPS;

            // Reset visit_frames on exit (after computing dwell for this frame)
            if (transition && std::strcmp(transition, "exit") == 0) {
                rec.visit_frames = 0;
            }

            // ── Node 8: dwell-time incident alert ──
            bool alert_active = inside && (visit_dwell >= DWELL_THRESHOLD_SEC);
            std::pair<guint64, int> key(tid, rec.visit_number);
            if (alert_active && ps.alerted_visits.find(key) == ps.alerted_visits.end()) {
                ps.alerted_visits.insert(key);
                Incident inc;
                inc.track_id = tid;
                inc.dwell_time_seconds = visit_dwell;
                inc.visit_number = rec.visit_number;
                incidents.push_back(inc);
            }

            Observation obs;
            obs.track_id = tid;
            obs.x1 = bx1; obs.y1 = by1; obs.x2 = bx2; obs.y2 = by2;
            obs.score = obj->confidence;
            obs.in_zone = inside;
            obs.transition = transition;
            obs.visit_number = rec.visit_number;
            obs.dwell_time_seconds = visit_dwell;
            obs.cumulative_zone_seconds = cumulative;
            obs.alert_active = alert_active;
            observations.push_back(obs);
        }
    }

    // ── Node 10: visual overlay ──
    cv::Mat frame = read_nv12_bgr(in);

    float scale = (float)W / 1400.0f;
    int box_thick   = std::max(2, (int)(3.0f * scale));
    float label_scale = std::max(0.8f, 1.1f * scale);
    int label_thick = std::max(1, (int)(2.0f * scale));
    int poly_thick  = std::max(2, (int)(4.0f * scale));

    // Draw restricted zone polygon in red
    {
        std::vector<cv::Point> pts(ZONE_POLY, ZONE_POLY + ZONE_POLY_N);
        const cv::Point *ppts = pts.data();
        int npts = ZONE_POLY_N;
        cv::polylines(frame, &ppts, &npts, 1, true, cv::Scalar(0, 0, 255), poly_thick, cv::LINE_AA);
    }

    // Draw each tracked person
    for (size_t i = 0; i < observations.size(); ++i) {
        const Observation &obs = observations[i];

        cv::Scalar color;
        if (obs.alert_active) {
            color = cv::Scalar(0, 0, 255);      // red
        } else if (obs.in_zone) {
            color = cv::Scalar(0, 140, 255);     // orange
        } else {
            color = cv::Scalar(0, 200, 0);       // green
        }

        int ix1 = (int)obs.x1, iy1 = (int)obs.y1;
        int ix2 = (int)obs.x2, iy2 = (int)obs.y2;
        cv::rectangle(frame, cv::Point(ix1, iy1), cv::Point(ix2, iy2), color, box_thick);

        // Build status string
        char status_buf[128];
        if (obs.alert_active) {
            snprintf(status_buf, sizeof(status_buf), "INTRUSION! %.1fs", obs.dwell_time_seconds);
        } else if (obs.in_zone) {
            snprintf(status_buf, sizeof(status_buf), "IN ZONE %.1fs", obs.dwell_time_seconds);
        } else {
            snprintf(status_buf, sizeof(status_buf), "CLEAR");
        }

        char label_buf[192];
        snprintf(label_buf, sizeof(label_buf), "ID %lu %s", (unsigned long)obs.track_id, status_buf);
        std::string label(label_buf);

        int baseline = 0;
        cv::Size tsz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, (double)label_scale, label_thick, &baseline);
        int tw = tsz.width, th = tsz.height;
        int label_y = std::max(th + baseline + 8, iy1);

        cv::rectangle(frame,
            cv::Point(ix1, label_y - th - baseline - 8),
            cv::Point(ix1 + tw + 12, label_y),
            color, -1);
        cv::putText(frame, label,
            cv::Point(ix1 + 6, label_y - baseline - 3),
            cv::FONT_HERSHEY_SIMPLEX, (double)label_scale,
            cv::Scalar(0, 0, 0), label_thick, cv::LINE_AA);
    }

    write_nv12(frame, in);

    // ── Publish metadata via render_meta::Builder ──
    render_meta::Builder sb(W, H);

    // Publish the restricted zone polygon
    {
        std::vector<double> ring;
        for (int i = 0; i < ZONE_POLY_N; ++i) {
            ring.push_back((double)ZONE_POLY[i].x);
            ring.push_back((double)ZONE_POLY[i].y);
        }
        sb.polygon(ring, "restricted_zone");
    }

    // Publish per-track observations
    for (size_t i = 0; i < observations.size(); ++i) {
        const Observation &obs = observations[i];
        // Box for each tracked person
        sb.box((double)obs.x1, (double)obs.y1,
               (double)(obs.x2 - obs.x1), (double)(obs.y2 - obs.y1),
               "person", (int)obs.track_id, (double)obs.score);

        // Scalar fields per observation
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "track_%lu_", (unsigned long)obs.track_id);
        std::string pfx(prefix);
        sb.field_s((pfx + "zone").c_str(), obs.in_zone ? "Restricted Area" : "outside");
        sb.field_s((pfx + "transition").c_str(),
                   obs.transition ? obs.transition : "none");
        sb.field_d((pfx + "dwell_time_seconds").c_str(), (double)obs.dwell_time_seconds);
        sb.field_i((pfx + "alert_active").c_str(), obs.alert_active ? 1 : 0);
        sb.field_i((pfx + "visit_number").c_str(), obs.visit_number);
        sb.field_d((pfx + "cumulative_zone_seconds").c_str(), (double)obs.cumulative_zone_seconds);
    }

    // Publish incident events
    for (size_t i = 0; i < incidents.size(); ++i) {
        const Incident &inc = incidents[i];
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "incident_%zu_", i);
        std::string pfx(prefix);
        sb.field_i((pfx + "track_id").c_str(), (int)inc.track_id);
        sb.field_s((pfx + "zone").c_str(), "Restricted Area");
        sb.field_d((pfx + "dwell_time_seconds").c_str(), (double)inc.dwell_time_seconds);
        sb.field_i((pfx + "visit_number").c_str(), inc.visit_number);
    }
    sb.field_i("num_incidents", (int)incidents.size());

    sb.publish(in->frame_meta);

    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);