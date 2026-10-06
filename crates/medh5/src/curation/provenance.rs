//! Provenance: a two-node W3C PROV-lite graph (spec §11.1).
//!
//! Agents do things; activities are the things done.  Objects point at an
//! activity through their `prov` attribute.  Two node types are enough to
//! describe the workflow that actually dominates curation --- a model
//! pre-annotates, a human corrects, a second human reviews --- which a "review
//! status" field cannot describe at all.

use indexmap::IndexMap;
use regex::Regex;
use serde_json::{json, Map, Value};
use std::sync::OnceLock;

use crate::json::{repr_list, repr_str};
use crate::pyval::{self, get_list, get_str, require, to_str};
use crate::{Error, Result};

/// `agent.type` values (§11.1).
pub const AGENT_TYPES: [&str; 4] = ["person", "software", "organization", "model"];
/// The schema's `agent` properties; the object is closed.
pub const AGENT_FIELDS: [&str; 7] = ["id", "type", "name", "role", "version", "qualification", "organization"];
/// The schema's `activity` properties; the object is closed.
pub const ACTIVITY_FIELDS: [&str; 9] =
    ["id", "type", "agent", "started", "ended", "tool", "inputs", "outputs", "params"];
/// `activity.type` values (§11.1).
pub const ACTIVITY_TYPES: [&str; 10] =
    ["import", "annotate", "review", "predict", "resample", "register", "derive", "deidentify", "transcode", "other"];

fn rfc3339() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})$").unwrap())
}

/// Whether `value` is an RFC 3339 timestamp, as the format checks it.
pub fn is_timestamp(value: &str) -> bool {
    rfc3339().is_match(value)
}

/// Validate an RFC 3339 timestamp (E604).
pub fn check_timestamp(value: &str, where_: &str) -> Result<()> {
    if !is_timestamp(value) {
        return Err(Error::coded("E604", format!("{where_}: {} is not an RFC 3339 timestamp", repr_str(value))));
    }
    Ok(())
}

/// Refuse a key the schema does not allow on a closed object (E005).
pub fn check_known(doc: &Map<String, Value>, known: &[&str], what: &str) -> Result<()> {
    let mut unknown: Vec<&String> = doc.keys().filter(|k| !known.contains(&k.as_str())).collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort();
    let mut known_sorted: Vec<&str> = known.to_vec();
    known_sorted.sort();
    Err(Error::coded(
        "E005",
        format!(
            "{what}: {} is not a {what} field; the schema closes this object, so it may hold only {}",
            repr_list(&unknown),
            repr_list(&known_sorted)
        ),
    ))
}

/// Someone or something that acts: a rater, a model, a tool, a site.
#[derive(Debug, Clone, PartialEq)]
pub struct Agent {
    pub id: String,
    pub r#type: String,
    pub name: String,
    pub role: Option<String>,
    pub version: Option<String>,
    pub qualification: Option<String>,
    /// The id of an `organization` agent this one belongs to (§11.1).
    pub organization: Option<String>,
}

impl Agent {
    /// A validated agent.
    pub fn new(id: impl Into<String>, agent_type: impl Into<String>, name: impl Into<String>) -> Result<Self> {
        let agent = Agent {
            id: id.into(),
            r#type: agent_type.into(),
            name: name.into(),
            role: None,
            version: None,
            qualification: None,
            organization: None,
        };
        agent.check()?;
        Ok(agent)
    }

    /// Validate the agent type (E603).
    pub fn check(&self) -> Result<()> {
        if !AGENT_TYPES.contains(&self.r#type.as_str()) {
            return Err(Error::coded(
                "E603",
                format!(
                    "agent {}: unknown type {}; expected one of {}",
                    repr_str(&self.id),
                    repr_str(&self.r#type),
                    repr_list(&AGENT_TYPES)
                ),
            ));
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.id));
        out.insert("type".into(), json!(self.r#type));
        out.insert("name".into(), json!(self.name));
        for (key, value) in [
            ("role", &self.role),
            ("version", &self.version),
            ("qualification", &self.qualification),
            ("organization", &self.organization),
        ] {
            if let Some(v) = value {
                out.insert(key.into(), json!(v));
            }
        }
        Value::Object(out)
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "an agent")?;
        check_known(doc, &AGENT_FIELDS, "agent")?;
        let agent = Agent {
            id: to_str(require(doc, "id")?),
            r#type: to_str(require(doc, "type")?),
            name: to_str(require(doc, "name")?),
            role: get_str(doc, "role"),
            version: get_str(doc, "version"),
            qualification: get_str(doc, "qualification"),
            organization: get_str(doc, "organization"),
        };
        agent.check()?;
        Ok(agent)
    }
}

/// One thing an agent did, with what it consumed and what it produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Activity {
    pub id: String,
    pub r#type: String,
    pub agent: Option<String>,
    pub started: Option<String>,
    pub ended: Option<String>,
    pub tool: Option<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub params: Map<String, Value>,
}

impl Activity {
    /// A validated activity with no optional fields.
    pub fn new(id: impl Into<String>, activity_type: impl Into<String>) -> Result<Self> {
        let activity = Activity {
            id: id.into(),
            r#type: activity_type.into(),
            agent: None,
            started: None,
            ended: None,
            tool: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            params: Map::new(),
        };
        activity.check()?;
        Ok(activity)
    }

    /// Validate the type (E603) and timestamps (E604).
    pub fn check(&self) -> Result<()> {
        if !ACTIVITY_TYPES.contains(&self.r#type.as_str()) {
            return Err(Error::coded(
                "E603",
                format!(
                    "activity {}: unknown type {}; expected one of {}",
                    repr_str(&self.id),
                    repr_str(&self.r#type),
                    repr_list(&ACTIVITY_TYPES)
                ),
            ));
        }
        for (key, value) in [("started", &self.started), ("ended", &self.ended)] {
            if let Some(v) = value {
                check_timestamp(v, &format!("activity {}.{key}", repr_str(&self.id)))?;
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.id));
        out.insert("type".into(), json!(self.r#type));
        for (key, value) in
            [("agent", &self.agent), ("started", &self.started), ("ended", &self.ended), ("tool", &self.tool)]
        {
            if let Some(v) = value {
                out.insert(key.into(), json!(v));
            }
        }
        if !self.inputs.is_empty() {
            out.insert("inputs".into(), json!(self.inputs));
        }
        if !self.outputs.is_empty() {
            out.insert("outputs".into(), json!(self.outputs));
        }
        if !self.params.is_empty() {
            out.insert("params".into(), Value::Object(self.params.clone()));
        }
        Value::Object(out)
    }

    pub fn from_json(doc: &Value) -> Result<Self> {
        let doc = pyval::as_object(doc, "an activity")?;
        check_known(doc, &ACTIVITY_FIELDS, "activity")?;
        let activity = Activity {
            id: to_str(require(doc, "id")?),
            r#type: to_str(require(doc, "type")?),
            agent: get_str(doc, "agent"),
            started: get_str(doc, "started"),
            ended: get_str(doc, "ended"),
            tool: get_str(doc, "tool"),
            inputs: get_list(doc, "inputs").iter().map(to_str).collect(),
            outputs: get_list(doc, "outputs").iter().map(to_str).collect(),
            params: match pyval::get(doc, "params") {
                Some(Value::Object(map)) => map.clone(),
                _ => Map::new(),
            },
        };
        activity.check()?;
        Ok(activity)
    }
}

/// The agents and activities of one sample, with reference resolution.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Provenance {
    agents: IndexMap<String, Agent>,
    activities: IndexMap<String, Activity>,
}

impl Provenance {
    /// A graph from nodes, refusing an id declared twice.
    pub fn new(agents: Vec<Agent>, activities: Vec<Activity>) -> Result<Self> {
        let mut prov = Provenance::default();
        for agent in agents {
            if prov.agents.contains_key(&agent.id) {
                return Err(duplicate("agent", &agent.id));
            }
            prov.agents.insert(agent.id.clone(), agent);
        }
        for activity in activities {
            if prov.activities.contains_key(&activity.id) {
                return Err(duplicate("activity", &activity.id));
            }
            prov.activities.insert(activity.id.clone(), activity);
        }
        Ok(prov)
    }

    /// Whether the graph holds any node.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty() && self.activities.is_empty()
    }

    pub fn agents(&self) -> impl Iterator<Item = &Agent> {
        self.agents.values()
    }

    pub fn activities(&self) -> impl Iterator<Item = &Activity> {
        self.activities.values()
    }

    pub fn n_agents(&self) -> usize {
        self.agents.len()
    }

    pub fn n_activities(&self) -> usize {
        self.activities.len()
    }

    /// An agent, or a `KeyError`.
    pub fn agent(&self, agent_id: &str) -> Result<&Agent> {
        self.agents.get(agent_id).ok_or_else(|| Error::Key(format!("unknown agent {}", repr_str(agent_id))))
    }

    /// An activity, or a `KeyError`.
    pub fn activity(&self, activity_id: &str) -> Result<&Activity> {
        self.activities
            .get(activity_id)
            .ok_or_else(|| Error::Key(format!("unknown activity {}", repr_str(activity_id))))
    }

    pub fn has_activity(&self, activity_id: &str) -> bool {
        self.activities.contains_key(activity_id)
    }

    pub fn has_agent(&self, agent_id: &str) -> bool {
        self.agents.contains_key(agent_id)
    }

    /// Add an agent; an id already in the graph is refused unless `replace`.
    pub fn add_agent(&mut self, agent: Agent, replace: bool) -> Result<Agent> {
        if !replace {
            if let Some(existing) = self.agents.get(&agent.id) {
                return Err(Error::invalid(format!(
                    "agent {} is already declared ({}); pass replace=True to rewrite it deliberately",
                    repr_str(&agent.id),
                    repr_str(&existing.name)
                )));
            }
        }
        self.agents.insert(agent.id.clone(), agent.clone());
        Ok(agent)
    }

    /// Add an activity; an id already in the graph is refused unless `replace`.
    pub fn add_activity(&mut self, activity: Activity, replace: bool) -> Result<Activity> {
        if !replace {
            if let Some(existing) = self.activities.get(&activity.id) {
                return Err(Error::invalid(format!(
                    "activity {} is already declared ({}); pass replace=True to rewrite it deliberately",
                    repr_str(&activity.id),
                    repr_str(&existing.r#type)
                )));
            }
        }
        self.activities.insert(activity.id.clone(), activity.clone());
        Ok(activity)
    }

    /// Every activity of one type.
    pub fn activities_by_type(&self, activity_type: &str) -> Vec<&Activity> {
        self.activities.values().filter(|a| a.r#type == activity_type).collect()
    }

    /// Every activity claiming `object_path` among its outputs.
    pub fn produced_by(&self, object_path: &str) -> Vec<&Activity> {
        self.activities.values().filter(|a| a.outputs.iter().any(|o| o == object_path)).collect()
    }

    /// `(activity_id, agent_id)` pairs whose agent is not declared (E605).
    pub fn dangling_agent_refs(&self) -> Vec<(String, String)> {
        self.activities
            .values()
            .filter_map(|a| match &a.agent {
                Some(agent) if !self.agents.contains_key(agent) => Some((a.id.clone(), agent.clone())),
                _ => None,
            })
            .collect()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "agents": self.agents.values().map(Agent::to_json).collect::<Vec<_>>(),
            "activities": self.activities.values().map(Activity::to_json).collect::<Vec<_>>(),
        })
    }

    pub fn from_json(doc: Option<&Value>) -> Result<Self> {
        let Some(doc) = doc else { return Ok(Provenance::default()) };
        if !pyval::truthy(doc) {
            return Ok(Provenance::default());
        }
        let doc = pyval::as_object(doc, "provenance")?;
        let agents = get_list(doc, "agents").iter().map(Agent::from_json).collect::<Result<Vec<_>>>()?;
        let activities = get_list(doc, "activities").iter().map(Activity::from_json).collect::<Result<Vec<_>>>()?;
        Provenance::new(agents, activities)
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        format!("Provenance({} agents, {} activities)", self.agents.len(), self.activities.len())
    }
}

fn duplicate(what: &str, id: &str) -> Error {
    Error::invalid(format!(
        "provenance declares {what} id {} more than once; a reference to it cannot say which one it means",
        repr_str(id)
    ))
}
