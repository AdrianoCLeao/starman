//! Localization (ADR 0015) on Project Fluent.
//!
//! Strings live in `assets/<root>/<locale>/*.ftl`. [`Localization`] loads
//! one bundle per supported locale, resolves messages through a fallback
//! chain (current locale → its language → the default locale), switches
//! locale at runtime (emitting [`LocaleChanged`]) and offers a pseudo
//! locale (`qps-ploc`) that accents and lengthens every string, to find
//! hard-coded or clipped text. [`check_directory`] powers
//! `starman l10n check`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use bevy_ecs::prelude::*;
use engine_core::{GameRuntime, RuntimePlugin};
use fluent_bundle::concurrent::FluentBundle;
use fluent_bundle::{FluentArgs, FluentResource, FluentValue};
use unic_langid::LanguageIdentifier;

pub use fluent_bundle;
pub use unic_langid;

/// The pseudo-localization locale id.
pub const PSEUDO_LOCALE: &str = "qps-ploc";

/// An argument value for a message.
#[derive(Clone, Debug, PartialEq)]
pub enum LocArg {
    Text(String),
    Number(f64),
}

impl From<&str> for LocArg {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for LocArg {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<f64> for LocArg {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<i64> for LocArg {
    fn from(value: i64) -> Self {
        Self::Number(value as f64)
    }
}

impl From<i32> for LocArg {
    fn from(value: i32) -> Self {
        Self::Number(value as f64)
    }
}

/// A problem found while loading or checking localization files.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LocIssue {
    pub locale: String,
    pub file: String,
    pub message: String,
}

impl std::fmt::Display for LocIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.locale, self.file, self.message)
    }
}

struct LocaleBundle {
    bundle: FluentBundle<FluentResource>,
    keys: BTreeSet<String>,
}

fn new_bundle(locale: &LanguageIdentifier) -> FluentBundle<FluentResource> {
    let mut bundle = FluentBundle::new_concurrent(vec![locale.clone()]);
    // Unicode isolation marks confuse fonts without bidi controls and
    // string comparisons in tests; the UI handles direction per run.
    bundle.set_use_isolating(false);
    bundle
}

/// Loaded strings and the active locale.
#[derive(Resource)]
pub struct Localization {
    bundles: HashMap<String, LocaleBundle>,
    supported: Vec<String>,
    default_locale: String,
    current: String,
    pseudo: bool,
    revision: u64,
    issues: Vec<LocIssue>,
    root: Option<PathBuf>,
}

impl Default for Localization {
    fn default() -> Self {
        Self {
            bundles: HashMap::new(),
            supported: vec!["en".to_owned()],
            default_locale: "en".to_owned(),
            current: "en".to_owned(),
            pseudo: false,
            revision: 1,
            issues: Vec::new(),
            root: None,
        }
    }
}

/// The active locale changed (UI re-resolves its strings).
#[derive(Event, Clone, Debug, PartialEq, Eq)]
pub struct LocaleChanged {
    pub locale: String,
}

fn parse_locale(locale: &str) -> LanguageIdentifier {
    locale
        .parse()
        .unwrap_or_else(|_| LanguageIdentifier::default())
}

/// Accents and lengthens `text` (outside `{placeables}`), keeping it
/// readable: "Save game" → "[Šàṽé ĝàɱé~~~~]".
pub fn pseudo_localize(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2 + 4);
    out.push('[');
    let mut letters: usize = 0;
    for c in text.chars() {
        let mapped = match c {
            'a' => 'à',
            'e' => 'é',
            'i' => 'î',
            'o' => 'ô',
            'u' => 'ü',
            'c' => 'ç',
            'n' => 'ñ',
            's' => 'š',
            'g' => 'ĝ',
            'm' => 'ɱ',
            'v' => 'ṽ',
            'A' => 'Å',
            'E' => 'É',
            'O' => 'Ö',
            'S' => 'Š',
            'U' => 'Û',
            other => other,
        };
        if c.is_alphabetic() {
            letters += 1;
        }
        out.push(mapped);
    }
    for _ in 0..letters.div_ceil(3) {
        out.push('~');
    }
    out.push(']');
    out
}

impl Localization {
    /// Loads `root/<locale>/*.ftl` for every supported locale. Problems
    /// (missing folders, parse errors, duplicate ids) are collected in
    /// [`Self::issues`]; loading never fails as a whole.
    pub fn load(root: impl Into<PathBuf>, default_locale: &str, supported: &[String]) -> Self {
        let root = root.into();
        let mut localization = Self {
            default_locale: default_locale.to_owned(),
            current: default_locale.to_owned(),
            root: Some(root.clone()),
            ..Default::default()
        };
        let mut locales: Vec<String> = supported.to_vec();
        if !locales.iter().any(|l| l == default_locale) {
            locales.insert(0, default_locale.to_owned());
        }
        for locale in &locales {
            let directory = root.join(locale);
            let mut files: Vec<PathBuf> = match std::fs::read_dir(&directory) {
                Ok(entries) => entries
                    .filter_map(|entry| entry.ok().map(|e| e.path()))
                    .filter(|path| path.extension().is_some_and(|ext| ext == "ftl"))
                    .collect(),
                Err(error) => {
                    localization.issues.push(LocIssue {
                        locale: locale.clone(),
                        file: directory.display().to_string(),
                        message: format!("cannot read locale folder: {error}"),
                    });
                    Vec::new()
                }
            };
            files.sort();
            let sources: Vec<(String, String)> = files
                .iter()
                .filter_map(|path| {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    match std::fs::read_to_string(path) {
                        Ok(text) => Some((name, text)),
                        Err(error) => {
                            localization.issues.push(LocIssue {
                                locale: locale.clone(),
                                file: name,
                                message: format!("unreadable: {error}"),
                            });
                            None
                        }
                    }
                })
                .collect();
            localization.add_locale(locale, &sources);
        }
        localization.supported = locales;
        localization
    }

    /// Builds a localization from in-memory `(locale, [(file, source)])`.
    pub fn from_sources(default_locale: &str, locales: &[(&str, Vec<(&str, &str)>)]) -> Self {
        let mut localization = Self {
            default_locale: default_locale.to_owned(),
            current: default_locale.to_owned(),
            supported: Vec::new(),
            ..Default::default()
        };
        for (locale, files) in locales {
            let files: Vec<(String, String)> = files
                .iter()
                .map(|(name, text)| (name.to_string(), text.to_string()))
                .collect();
            localization.add_locale(locale, &files);
            localization.supported.push(locale.to_string());
        }
        localization
    }

    fn add_locale(&mut self, locale: &str, files: &[(String, String)]) {
        let id = parse_locale(locale);
        let mut bundle = new_bundle(&id);
        let mut keys = BTreeSet::new();
        for (file, source) in files {
            let resource = match FluentResource::try_new(source.clone()) {
                Ok(resource) => resource,
                Err((resource, errors)) => {
                    for error in errors {
                        self.issues.push(LocIssue {
                            locale: locale.to_owned(),
                            file: file.clone(),
                            message: format!("parse error: {error:?}"),
                        });
                    }
                    resource
                }
            };
            for entry in resource.entries() {
                if let fluent_syntax::ast::Entry::Message(message) = entry {
                    keys.insert(message.id.name.to_owned());
                    for attribute in &message.attributes {
                        keys.insert(format!("{}.{}", message.id.name, attribute.id.name));
                    }
                }
            }
            if let Err(errors) = bundle.add_resource(resource) {
                for error in errors {
                    self.issues.push(LocIssue {
                        locale: locale.to_owned(),
                        file: file.clone(),
                        message: format!("{error}"),
                    });
                }
            }
        }
        self.bundles
            .insert(locale.to_owned(), LocaleBundle { bundle, keys });
    }

    /// Reloads every locale from disk (editor, hot reload).
    pub fn reload(&mut self) {
        let Some(root) = self.root.clone() else {
            return;
        };
        let current = self.current.clone();
        let pseudo = self.pseudo;
        let revision = self.revision;
        *self = Self::load(root, &self.default_locale.clone(), &self.supported.clone());
        self.current = current;
        self.pseudo = pseudo;
        self.revision = revision + 1;
    }

    pub fn issues(&self) -> &[LocIssue] {
        &self.issues
    }

    pub fn supported(&self) -> &[String] {
        &self.supported
    }

    pub fn default_locale(&self) -> &str {
        &self.default_locale
    }

    pub fn current(&self) -> &str {
        if self.pseudo {
            PSEUDO_LOCALE
        } else {
            &self.current
        }
    }

    /// Bumped on every locale or content change.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Switches locale (`qps-ploc` enables pseudo-localization on top of
    /// the default locale). Returns `false` for unsupported locales.
    pub fn set_locale(&mut self, locale: &str) -> bool {
        if locale == PSEUDO_LOCALE {
            if !self.pseudo {
                self.pseudo = true;
                self.revision += 1;
            }
            return true;
        }
        if !self.bundles.contains_key(locale) {
            return false;
        }
        if self.current != locale || self.pseudo {
            self.current = locale.to_owned();
            self.pseudo = false;
            self.revision += 1;
        }
        true
    }

    /// Locales consulted for a lookup, most specific first.
    fn chain(&self) -> Vec<&str> {
        let base = if self.pseudo {
            self.default_locale.as_str()
        } else {
            self.current.as_str()
        };
        let mut chain = vec![base];
        if let Some((language, _)) = base.split_once('-') {
            if self.bundles.contains_key(language) {
                chain.push(language);
            }
        }
        if !chain.contains(&self.default_locale.as_str()) {
            chain.push(&self.default_locale);
        }
        chain
    }

    pub fn has(&self, key: &str) -> bool {
        self.chain().iter().any(|locale| {
            self.bundles
                .get(*locale)
                .is_some_and(|b| b.keys.contains(key))
        })
    }

    /// The message `key` (`id` or `id.attribute`), or `key` itself when
    /// missing (so gaps are visible, never blank).
    pub fn tr(&self, key: &str) -> String {
        self.tr_args(key, &[])
    }

    pub fn tr_args(&self, key: &str, args: &[(&str, LocArg)]) -> String {
        let (id, attribute) = match key.split_once('.') {
            Some((id, attribute)) => (id, Some(attribute)),
            None => (key, None),
        };
        let mut fluent_args = FluentArgs::new();
        for (name, value) in args {
            match value {
                LocArg::Text(text) => fluent_args.set(*name, FluentValue::from(text.clone())),
                LocArg::Number(number) => fluent_args.set(*name, FluentValue::from(*number)),
            }
        }
        for locale in self.chain() {
            let Some(locale_bundle) = self.bundles.get(locale) else {
                continue;
            };
            let bundle = &locale_bundle.bundle;
            let Some(message) = bundle.get_message(id) else {
                continue;
            };
            let pattern = match attribute {
                Some(attribute) => message.get_attribute(attribute).map(|a| a.value()),
                None => message.value(),
            };
            let Some(pattern) = pattern else {
                continue;
            };
            let mut errors = Vec::new();
            let text = bundle
                .format_pattern(pattern, Some(&fluent_args), &mut errors)
                .into_owned();
            if !errors.is_empty() {
                log::debug!(target: "engine::l10n", "formatting '{key}': {errors:?}");
            }
            return if self.pseudo {
                pseudo_localize(&text)
            } else {
                text
            };
        }
        key.to_owned()
    }

    /// Keys per locale (for tools).
    pub fn keys(&self, locale: &str) -> Option<&BTreeSet<String>> {
        self.bundles.get(locale).map(|b| &b.keys)
    }

    /// Keys of the default locale missing in other locales, and keys other
    /// locales define that the default locale lacks.
    pub fn report(&self) -> LocalizationReport {
        let mut report = LocalizationReport {
            issues: self.issues.clone(),
            ..Default::default()
        };
        let reference = self
            .bundles
            .get(&self.default_locale)
            .map(|b| b.keys.clone())
            .unwrap_or_default();
        for locale in &self.supported {
            if *locale == self.default_locale {
                continue;
            }
            let keys = self
                .bundles
                .get(locale)
                .map(|b| b.keys.clone())
                .unwrap_or_default();
            let missing: Vec<String> = reference.difference(&keys).cloned().collect();
            let extra: Vec<String> = keys.difference(&reference).cloned().collect();
            if !missing.is_empty() {
                report.missing.insert(locale.clone(), missing);
            }
            if !extra.is_empty() {
                report.extra.insert(locale.clone(), extra);
            }
        }
        report
    }
}

/// Result of a localization check.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocalizationReport {
    pub issues: Vec<LocIssue>,
    /// Locale → keys present in the default locale but missing here.
    pub missing: BTreeMap<String, Vec<String>>,
    /// Locale → keys not present in the default locale.
    pub extra: BTreeMap<String, Vec<String>>,
}

impl LocalizationReport {
    /// Errors (parse problems, missing keys) make the check fail; extra
    /// keys are warnings.
    pub fn is_ok(&self) -> bool {
        self.issues.is_empty() && self.missing.is_empty()
    }
}

/// Loads and checks a localization folder (`starman l10n check`).
pub fn check_directory(
    root: &Path,
    default_locale: &str,
    supported: &[String],
) -> LocalizationReport {
    Localization::load(root, default_locale, supported).report()
}

/// Installs the [`Localization`] resource (empty until a host loads the
/// project strings) and the [`LocaleChanged`] event.
#[derive(Default)]
pub struct LocalizationPlugin;

impl RuntimePlugin for LocalizationPlugin {
    fn name(&self) -> &'static str {
        "engine::localization"
    }

    fn build(&self, runtime: &mut GameRuntime) {
        runtime
            .init_resource::<Localization>()
            .add_event::<LocaleChanged>()
            .add_systems(engine_core::ScheduleKind::First, announce_locale_changes);
    }
}

fn announce_locale_changes(
    localization: Option<Res<Localization>>,
    mut last: Local<Option<(u64, String)>>,
    mut events: EventWriter<LocaleChanged>,
) {
    let Some(localization) = localization else {
        return;
    };
    let current = localization.current().to_owned();
    let changed = match last.as_ref() {
        Some((_, locale)) => *locale != current,
        None => false,
    };
    if changed {
        events.send(LocaleChanged {
            locale: current.clone(),
        });
    }
    *last = Some((localization.revision(), current));
}

#[cfg(test)]
mod tests {
    use super::*;

    const EN: &str = "
hud-keys = Keys: { $count }
menu-save = Save game
    .tooltip = Write the current state to a slot
items-found = { $count ->
    [one] Found one item
   *[other] Found { $count } items
}
only-english = Only in English
";

    const PT: &str = "
hud-keys = Chaves: { $count }
menu-save = Salvar jogo
    .tooltip = Grava o estado atual em um slot
items-found = { $count ->
    [one] Encontrou um item
   *[other] Encontrou { $count } itens
}
only-portuguese = Só em português
";

    fn sample() -> Localization {
        Localization::from_sources(
            "en",
            &[
                ("en", vec![("main.ftl", EN)]),
                ("pt-BR", vec![("main.ftl", PT)]),
            ],
        )
    }

    #[test]
    fn resolves_arguments_attributes_plurals_and_fallback() {
        let mut l10n = sample();
        assert_eq!(l10n.tr_args("hud-keys", &[("count", 2.into())]), "Keys: 2");
        assert_eq!(
            l10n.tr("menu-save.tooltip"),
            "Write the current state to a slot"
        );
        assert_eq!(
            l10n.tr_args("items-found", &[("count", 1.into())]),
            "Found one item"
        );
        assert_eq!(
            l10n.tr_args("items-found", &[("count", 3.into())]),
            "Found 3 items"
        );

        let revision = l10n.revision();
        assert!(l10n.set_locale("pt-BR"));
        assert!(l10n.revision() > revision);
        assert_eq!(l10n.tr("menu-save"), "Salvar jogo");
        assert_eq!(
            l10n.tr_args("items-found", &[("count", 3.into())]),
            "Encontrou 3 itens"
        );
        assert_eq!(
            l10n.tr("only-english"),
            "Only in English",
            "falls back to the default locale"
        );
        assert_eq!(l10n.tr("does-not-exist"), "does-not-exist");
        assert!(!l10n.set_locale("fr"));
    }

    #[test]
    fn pseudo_locale_accents_and_lengthens() {
        let mut l10n = sample();
        assert!(l10n.set_locale(PSEUDO_LOCALE));
        let text = l10n.tr("menu-save");
        assert!(text.starts_with("[Šàṽé ĝàɱé"), "{text}");
        assert!(text.chars().count() > "Save game".len() + 2);
        assert_eq!(l10n.current(), PSEUDO_LOCALE);
    }

    #[test]
    fn report_lists_missing_and_extra_keys_and_parse_errors() {
        let broken = Localization::from_sources(
            "en",
            &[
                ("en", vec![("main.ftl", EN)]),
                ("pt-BR", vec![("main.ftl", PT), ("bad.ftl", "= nope")]),
            ],
        );
        let report = broken.report();
        assert_eq!(report.missing["pt-BR"], vec!["only-english".to_owned()]);
        assert_eq!(report.extra["pt-BR"], vec!["only-portuguese".to_owned()]);
        assert!(
            report.issues.iter().any(|i| i.file == "bad.ftl"),
            "{:?}",
            report.issues
        );
        assert!(!report.is_ok());
    }

    #[test]
    fn loads_locale_folders_from_disk() {
        let root = std::env::temp_dir().join(format!(
            "starman-l10n-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for (locale, text) in [("en", EN), ("pt-BR", PT)] {
            std::fs::create_dir_all(root.join(locale)).unwrap();
            std::fs::write(root.join(locale).join("main.ftl"), text).unwrap();
        }
        let mut l10n = Localization::load(&root, "en", &["en".into(), "pt-BR".into(), "fr".into()]);
        assert!(
            l10n.issues().iter().any(|i| i.locale == "fr"),
            "missing folder reported"
        );
        assert!(l10n.set_locale("pt-BR"));
        std::fs::write(root.join("pt-BR").join("main.ftl"), "menu-save = Gravar\n").unwrap();
        l10n.reload();
        assert_eq!(l10n.current(), "pt-BR");
        assert_eq!(l10n.tr("menu-save"), "Gravar");
        let _ = std::fs::remove_dir_all(&root);
    }
}
