// SPDX-License-Identifier: Apache-2.0

//! Evidence for default attributes when HGPTI's driver cannot query a node.
//! Only freshly observed streams and complete single-stream captures qualify.

use super::{ExportResult, invalid};
use std::collections::{BTreeMap, BTreeSet};

pub(super) const OPTIONAL: [i32; 7] = [5, 8, 9, 10, 11, 13, 14];
pub(super) const CALLBACKS: &[(u32, &str)] = &[
    (124, "cuStreamCreate"),
    (367, "cuStreamCreateWithPriority"),
    (127, "cuStreamDestroy"),
    (326, "cuStreamDestroy_v2"),
    (495, "cuStreamBeginCapture"),
    (496, "cuStreamBeginCapture_ptsz"),
    (539, "cuStreamBeginCapture_v2"),
    (540, "cuStreamBeginCapture_v2_ptsz"),
    (497, "cuStreamEndCapture"),
    (498, "cuStreamEndCapture_ptsz"),
    (720, "cuStreamBeginCaptureToGraph"),
    (721, "cuStreamBeginCaptureToGraph_ptsz"),
    (517, "cuGraphDestroy"),
    (571, "cuGraphKernelNodeSetAttribute"),
    (569, "cuGraphKernelNodeCopyAttributes"),
    (576, "cuStreamSetAttribute"),
    (577, "cuStreamSetAttribute_ptsz"),
    (572, "cuStreamCopyAttributes"),
    (573, "cuStreamCopyAttributes_ptsz"),
];

#[repr(C)]
struct StreamPriority {
    output: *const usize,
    flags: u32,
    priority: i32,
}
#[repr(C)]
struct Launch {
    function: usize,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    stream: usize,
    arguments: usize,
    extra: usize,
}
#[repr(C)]
struct Config {
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    stream: usize,
    attributes: *const Attribute,
    count: u32,
}
#[repr(C)]
struct Attribute {
    id: i32,
    value: [u64; 8],
}

#[derive(Clone, Default)]
struct Capture {
    context: usize,
    complete: bool,
    explicit: BTreeSet<i32>,
    functions: BTreeMap<usize, usize>,
}

#[derive(Default)]
pub(super) struct History {
    streams: BTreeMap<usize, (usize, BTreeSet<i32>)>,
    captures: BTreeMap<usize, Capture>,
    graphs: BTreeMap<usize, Capture>,
    changes: BTreeMap<usize, BTreeSet<i32>>,
}

pub(super) struct Defaults {
    explicit: BTreeSet<i32>,
    nodes: BTreeMap<usize, BTreeSet<i32>>,
}
impl Defaults {
    pub(super) fn check(&self, node: usize, id: i32) -> ExportResult<()> {
        if !OPTIONAL.contains(&id)
            || self.explicit.contains(&id)
            || self.nodes.get(&node).is_some_and(|ids| ids.contains(&id))
        {
            return Err(invalid(format!(
                "graph attribute {id} was explicitly configured or copied; its driver getter must work"
            )));
        }
        Ok(())
    }
}

pub(super) fn omit_unavailable(
    status: i32,
    id: i32,
    node: usize,
    defaults: Option<&ExportResult<Defaults>>,
) -> ExportResult<bool> {
    if status != 1 {
        return Ok(false);
    }
    if let Some(defaults) = defaults {
        if OPTIONAL.contains(&id) {
            let defaults = defaults.as_ref().map_err(|error| invalid(format!(
                "cuGraphKernelNodeGetAttribute({id}) is unavailable and its default is unproven: {error}"
            )))?;
            defaults.check(node, id)?;
            return Ok(true);
        }
    } else if id == 11 {
        // Existing CUDA compatibility for the newer preferred-cluster query.
        return Ok(true);
    }
    Ok(false)
}

pub(super) fn is_launch(name: &str) -> bool {
    matches!(
        name,
        "cuLaunchKernel"
            | "cuLaunchKernel_ptsz"
            | "cuLaunchKernelEx"
            | "cuLaunchKernelEx_ptsz"
            | "cuLaunchCooperativeKernel"
            | "cuLaunchCooperativeKernel_ptsz"
    )
}

pub(super) fn capture_boundary(name: &str) -> bool {
    name.starts_with("cuStreamBeginCapture") || name.starts_with("cuStreamEndCapture")
}

impl History {
    pub(super) fn failed(&mut self, name: &str, stream: usize) {
        if name.starts_with("cuStreamEndCapture") {
            self.captures.remove(&stream);
        }
    }
    pub(super) fn drop_context(&mut self, context: usize) {
        self.streams
            .retain(|_, (ctx, _)| *ctx != 0 && *ctx != context);
        self.captures
            .retain(|_, capture| capture.context != 0 && capture.context != context);
        self.graphs
            .retain(|_, capture| capture.context != 0 && capture.context != context);
    }

    pub(super) fn defaults(
        &self,
        graph: usize,
        functions: &BTreeMap<usize, usize>,
    ) -> ExportResult<Defaults> {
        let capture = self.graphs.get(&graph).ok_or_else(|| invalid(
            "graph capture was not completely observed by HGPTI; use a driver with working attribute queries",
        ))?;
        if !capture.complete || &capture.functions != functions {
            return Err(invalid(
                "cannot prove graph attribute defaults: require a freshly recorded stream, a complete \
                 single-stream capture and an exact kernel launch inventory",
            ));
        }
        Ok(Defaults {
            explicit: capture.explicit.clone(),
            nodes: self.changes.clone(),
        })
    }

    // Called on successful API EXIT only, with parameters owned by CUPTI.
    // Never query CUDA from a callback or read undocumented driver objects.
    pub(super) unsafe fn observe(
        &mut self,
        name: &str,
        context: usize,
        params: *const usize,
        nested_launch: bool,
    ) -> ExportResult<()> {
        let value = |index| unsafe { *params.add(index) };
        let output = |index| -> ExportResult<usize> {
            let pointer = value(index) as *const usize;
            if pointer.is_null() {
                return Err(invalid("null graph-history callback output"));
            }
            Ok(unsafe { *pointer })
        };
        match name {
            "cuStreamCreate" | "cuStreamCreateWithPriority" => {
                let mut explicit = BTreeSet::new();
                if name == "cuStreamCreateWithPriority"
                    && unsafe { &*params.cast::<StreamPriority>() }.priority != 0
                {
                    explicit.insert(8);
                }
                self.streams.insert(output(0)?, (context, explicit));
            }
            "cuStreamDestroy" | "cuStreamDestroy_v2" => {
                self.streams.remove(&value(0));
                self.captures.remove(&value(0));
            }
            "cuStreamBeginCapture"
            | "cuStreamBeginCapture_ptsz"
            | "cuStreamBeginCapture_v2"
            | "cuStreamBeginCapture_v2_ptsz" => {
                let stream = self.streams.get(&value(0));
                self.captures.insert(
                    value(0),
                    Capture {
                        context: if context != 0 {
                            context
                        } else {
                            stream.map_or(0, |s| s.0)
                        },
                        complete: value(0) > 2 && stream.is_some(),
                        explicit: stream.map(|s| s.1.clone()).unwrap_or_default(),
                        ..Default::default()
                    },
                );
            }
            "cuStreamEndCapture" | "cuStreamEndCapture_ptsz" => {
                let graph = output(1)?;
                self.graphs.remove(&graph);
                if let Some(capture) = self.captures.remove(&value(0)) {
                    self.graphs.insert(graph, capture);
                }
            }
            "cuStreamBeginCaptureToGraph" | "cuStreamBeginCaptureToGraph_ptsz" => {
                self.graphs.remove(&value(1));
                self.captures.remove(&value(0));
            }
            "cuGraphDestroy" => {
                self.graphs.remove(&value(0));
            }
            "cuFuncSetAttribute" | "cuKernelSetAttribute" => {
                // A default node can inherit function policy. If that policy
                // changes after capture, the export-time function snapshot no
                // longer proves the captured node's policy.
                let kernel = name == "cuKernelSetAttribute";
                let attribute = value(if kernel { 0 } else { 1 }) as i32;
                if let Some(id) = match attribute {
                    15 => Some(5),
                    9 => Some(14),
                    _ => None,
                } {
                    for capture in self.captures.values_mut() {
                        if context == 0 || capture.context == 0 || capture.context == context {
                            capture.explicit.insert(id);
                        }
                    }
                    for capture in self.graphs.values_mut() {
                        if (!kernel && capture.functions.contains_key(&value(0)))
                            || (kernel
                                && (context == 0
                                    || capture.context == 0
                                    || capture.context == context))
                        {
                            capture.explicit.insert(id);
                        }
                    }
                }
            }
            "cuGraphKernelNodeSetAttribute" => {
                self.changes
                    .entry(value(0))
                    .or_default()
                    .insert(value(1) as i32);
            }
            "cuGraphKernelNodeCopyAttributes" => {
                self.changes.entry(value(0)).or_default().extend(OPTIONAL);
            }
            "cuStreamSetAttribute"
            | "cuStreamSetAttribute_ptsz"
            | "cuStreamCopyAttributes"
            | "cuStreamCopyAttributes_ptsz" => {
                let ids: Vec<_> = if name.starts_with("cuStreamSet") {
                    vec![value(1) as i32]
                } else {
                    OPTIONAL.to_vec()
                };
                if let Some((_, explicit)) = self.streams.get_mut(&value(0)) {
                    explicit.extend(&ids);
                }
                for (owner_stream, capture) in &mut self.captures {
                    if *owner_stream == value(0) {
                        capture.explicit.extend(&ids);
                    }
                }
            }
            name if is_launch(name) => {
                if self.captures.is_empty() {
                    return Ok(());
                }
                let mut explicit = BTreeSet::new();
                let (function, stream) = if name.starts_with("cuLaunchKernelEx") {
                    let config = value(0) as *const Config;
                    if config.is_null() {
                        return Err(invalid("null extended launch configuration"));
                    }
                    let config = unsafe { &*config };
                    if config.count > 64 || (config.count != 0 && config.attributes.is_null()) {
                        return Err(invalid("invalid extended launch attribute list"));
                    }
                    for index in 0..config.count as usize {
                        let id = unsafe { (*config.attributes.add(index)).id };
                        if OPTIONAL.contains(&id) {
                            explicit.insert(id);
                        } else if ![0, 1, 2, 4].contains(&id) {
                            explicit.extend(OPTIONAL);
                        }
                    }
                    (value(1), config.stream)
                } else {
                    let launch = unsafe { &*params.cast::<Launch>() };
                    (launch.function, launch.stream)
                };
                for (&owner_stream, capture) in &mut self.captures {
                    if stream != owner_stream {
                        if context == 0 || capture.context == 0 || context == capture.context {
                            capture.complete = false;
                        }
                        continue;
                    }
                    if capture.context == 0 {
                        capture.context = context;
                    }
                    capture.explicit.extend(&explicit);
                    if !nested_launch {
                        *capture.functions.entry(function).or_default() += 1;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const CONTEXT: usize = 17;
    const STREAM: usize = 19;
    const GRAPH: usize = 23;
    const FUNCTION: usize = 29;
    fn api(history: &mut History, name: &str, args: &[usize]) {
        unsafe {
            history
                .observe(name, CONTEXT, args.as_ptr(), false)
                .unwrap();
        }
    }
    fn begin(history: &mut History, fresh: bool) {
        let stream = STREAM;
        if fresh {
            api(
                history,
                "cuStreamCreate",
                &[&stream as *const _ as usize, 0],
            );
        }
        api(history, "cuStreamBeginCapture_v2", &[STREAM, 0]);
    }
    fn launch(history: &mut History, stream: usize, attribute: Option<i32>) {
        let attrs = Attribute {
            id: attribute.unwrap_or(0),
            value: [0; 8],
        };
        let config = Config {
            grid: [1; 3],
            block: [1; 3],
            shared: 0,
            stream,
            attributes: &attrs,
            count: u32::from(attribute.is_some()),
        };
        api(
            history,
            "cuLaunchKernelEx",
            &[&config as *const _ as usize, FUNCTION, 0, 0],
        );
    }
    fn end(history: &mut History) {
        let graph = GRAPH;
        api(
            history,
            "cuStreamEndCapture",
            &[STREAM, &graph as *const _ as usize],
        );
    }
    fn defaults(history: &History) -> ExportResult<Defaults> {
        history.defaults(GRAPH, &[(FUNCTION, 1)].into())
    }
    fn captured(attribute: Option<i32>) -> History {
        let mut history = History::default();
        begin(&mut history, true);
        launch(&mut history, STREAM, attribute);
        end(&mut history);
        history
    }
    #[test]
    fn nested_launches_retain_attributes_without_double_counting_kernels() {
        let mut history = History::default();
        begin(&mut history, true);
        let attribute = Attribute {
            id: 5,
            value: [0; 8],
        };
        let config = Config {
            grid: [1; 3],
            block: [1; 3],
            shared: 0,
            stream: STREAM,
            attributes: &attribute,
            count: 1,
        };
        let arguments = [&config as *const _ as usize, FUNCTION, 0, 0];
        unsafe {
            history
                .observe("cuLaunchKernelEx", CONTEXT, arguments.as_ptr(), true)
                .unwrap();
        }
        launch(&mut history, STREAM, None);
        end(&mut history);
        assert!(defaults(&history).unwrap().check(100, 5).is_err());
    }
    #[test]
    fn changed_inherited_function_policy_requires_a_readable_node_attribute() {
        for (attribute, id) in [(15, 5), (9, 14)] {
            let mut history = captured(None);
            api(&mut history, "cuFuncSetAttribute", &[FUNCTION, attribute]);
            assert!(defaults(&history).unwrap().check(100, id).is_err());
            history = captured(None);
            api(&mut history, "cuKernelSetAttribute", &[attribute, 0xabc, 0]);
            assert!(defaults(&history).unwrap().check(100, id).is_err());
            history = captured(None);
            api(
                &mut history,
                "cuFuncSetAttribute",
                &[FUNCTION + 1, attribute],
            );
            assert!(defaults(&history).unwrap().check(100, id).is_ok());
        }
    }
    #[test]
    fn contextless_stream_and_graph_callbacks_do_not_lose_mutations() {
        let mut history = History::default();
        let stream = STREAM;
        unsafe {
            history
                .observe(
                    "cuStreamCreate",
                    0,
                    [&stream as *const _ as usize, 0].as_ptr(),
                    false,
                )
                .unwrap();
        }
        begin(&mut history, false);
        unsafe {
            history
                .observe("cuStreamSetAttribute", 0, [STREAM, 9, 0].as_ptr(), false)
                .unwrap();
        }
        launch(&mut history, STREAM, None);
        end(&mut history);
        assert!(defaults(&history).unwrap().check(100, 9).is_err());
        unsafe {
            history
                .observe(
                    "cuGraphKernelNodeSetAttribute",
                    0,
                    [100, 5, 0].as_ptr(),
                    false,
                )
                .unwrap();
        }
        assert!(defaults(&history).unwrap().check(100, 5).is_err());
    }
    #[test]
    fn only_proven_optional_defaults_can_omit_invalid_value_queries() {
        let history = captured(None);
        let proof = defaults(&history);
        for id in OPTIONAL {
            assert!(omit_unavailable(1, id, 100, Some(&proof)).unwrap());
            for status in [0, 3, 400, 801, 999] {
                assert!(!omit_unavailable(status, id, 100, Some(&proof)).unwrap());
            }
        }
        for id in [1, 2, 4, 99] {
            assert!(!omit_unavailable(1, id, 100, Some(&proof)).unwrap());
        }
        assert!(!omit_unavailable(1, 5, 100, None).unwrap());
        assert!(omit_unavailable(1, 11, 100, None).unwrap());
    }
    #[test]
    fn configured_launch_node_and_copied_attributes_are_never_assumed_default() {
        for id in OPTIONAL {
            let mut history = captured(Some(id));
            assert!(omit_unavailable(1, id, 100, Some(&defaults(&history))).is_err());
            // A supported getter remains authoritative even when proof is unavailable.
            assert!(!omit_unavailable(0, id, 100, Some(&defaults(&history))).unwrap());
            history = captured(None);
            api(
                &mut history,
                "cuGraphKernelNodeSetAttribute",
                &[100, id as usize, 0],
            );
            assert!(defaults(&history).unwrap().check(100, id).is_err());
            assert!(defaults(&history).unwrap().check(101, id).is_ok());
            api(&mut history, "cuGraphKernelNodeCopyAttributes", &[101, 100]);
            assert!(defaults(&history).unwrap().check(101, id).is_err());
        }
        assert!(
            defaults(&captured(Some(18)))
                .unwrap()
                .check(100, 5)
                .is_err()
        );
    }
    #[test]
    fn incomplete_captures_preexisting_streams_other_streams_and_inventory_changes_fail() {
        let mut history = History::default();
        begin(&mut history, false);
        launch(&mut history, STREAM, None);
        end(&mut history);
        assert!(defaults(&history).is_err());
        history = History::default();
        begin(&mut history, true);
        launch(&mut history, STREAM + 1, None);
        end(&mut history);
        assert!(defaults(&history).is_err());
        history = captured(None);
        assert!(history.defaults(GRAPH, &BTreeMap::new()).is_err());
        assert!(
            history
                .defaults(GRAPH, &[(FUNCTION + 1, 1)].into())
                .is_err()
        );
        assert!(history.defaults(GRAPH, &[(FUNCTION, 2)].into()).is_err());
        assert!(
            history
                .defaults(GRAPH + 1, &[(FUNCTION, 1)].into())
                .is_err()
        );
    }
    #[test]
    fn stream_attributes_and_priority_require_readable_node_values() {
        for name in ["cuStreamSetAttribute", "cuStreamCopyAttributes"] {
            let mut history = History::default();
            begin(&mut history, true);
            api(&mut history, name, &[STREAM, 9, 0]);
            launch(&mut history, STREAM, None);
            end(&mut history);
            assert!(defaults(&history).unwrap().check(100, 9).is_err());
        }
        let mut history = History::default();
        let stream = STREAM;
        let params = StreamPriority {
            output: &stream,
            flags: 0,
            priority: -1,
        };
        unsafe {
            history
                .observe(
                    "cuStreamCreateWithPriority",
                    CONTEXT,
                    (&params as *const StreamPriority).cast(),
                    false,
                )
                .unwrap();
        }
        begin(&mut history, false);
        launch(&mut history, STREAM, None);
        end(&mut history);
        assert!(defaults(&history).unwrap().check(100, 8).is_err());
    }
    #[test]
    fn destroyed_contexts_graphs_failed_captures_and_capture_to_existing_graph_invalidate_proof() {
        let mut history = captured(None);
        api(&mut history, "cuGraphDestroy", &[GRAPH]);
        assert!(defaults(&history).is_err());
        history = captured(None);
        history.drop_context(CONTEXT);
        assert!(defaults(&history).is_err());
        history = History::default();
        begin(&mut history, true);
        history.failed("cuStreamEndCapture", STREAM);
        end(&mut history);
        assert!(defaults(&history).is_err());
        history = captured(None);
        api(
            &mut history,
            "cuStreamBeginCaptureToGraph",
            &[STREAM, GRAPH],
        );
        assert!(defaults(&history).is_err());
        end(&mut history);
        assert!(defaults(&history).is_err());
    }
}
