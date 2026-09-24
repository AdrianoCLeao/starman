use std::fmt::Write;
use std::path::PathBuf;

use engine_localization::check_directory;
use engine_project::Project;

use crate::error::CliError;

/// `starman l10n check`: parses every locale of the project and compares
/// keys against the default locale.
pub fn check(path: PathBuf) -> Result<String, CliError> {
    let project = Project::open(&path).map_err(|source| CliError::Engine {
        path: path.clone(),
        source,
    })?;
    let settings = &project.manifest.game.localization;
    let root = project.paths.assets_dir().join(&settings.root);
    let report = check_directory(&root, &settings.default_locale, &settings.supported);

    let mut text = String::new();
    for issue in &report.issues {
        let _ = writeln!(text, "error: {issue}");
    }
    for (locale, keys) in &report.missing {
        let _ = writeln!(
            text,
            "error: [{locale}] missing {} key(s): {}",
            keys.len(),
            keys.join(", ")
        );
    }
    for (locale, keys) in &report.extra {
        let _ = writeln!(
            text,
            "warning: [{locale}] {} key(s) not in '{}': {}",
            keys.len(),
            settings.default_locale,
            keys.join(", ")
        );
    }
    if report.is_ok() {
        Ok(format!(
            "Localization OK: {} locale(s) under '{}'.{}{}",
            settings.supported.len(),
            root.display(),
            if text.is_empty() { "" } else { "\n" },
            text.trim_end()
        ))
    } else {
        Err(CliError::Check {
            path,
            report: text.trim_end().to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_project::CreateOptions;

    #[test]
    fn reports_missing_keys_and_succeeds_once_fixed() {
        let root = std::env::temp_dir().join(format!(
            "starman-cli-l10n-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let project = Project::create(
            &root,
            CreateOptions {
                name: "L10n".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let mut manifest = project.manifest.clone();
        manifest.game.localization.supported = vec!["en".into(), "pt-BR".into()];
        let mut project = project;
        project.manifest = manifest;
        project.save_manifest().unwrap();
        let dir = project
            .paths
            .assets_dir()
            .join(&project.manifest.game.localization.root);
        std::fs::create_dir_all(dir.join("en")).unwrap();
        std::fs::create_dir_all(dir.join("pt-BR")).unwrap();
        std::fs::write(dir.join("en/main.ftl"), "a = A\nb = B\n").unwrap();
        std::fs::write(dir.join("pt-BR/main.ftl"), "a = A\nz = Z\n").unwrap();
        let error = check(root.clone()).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("missing 1 key(s): b"), "{text}");
        assert!(text.contains("not in 'en': z"), "{text}");
        assert_eq!(error.exit_code(), 5);
        std::fs::write(dir.join("pt-BR/main.ftl"), "a = A\nb = B\n").unwrap();
        assert!(check(root.clone()).unwrap().starts_with("Localization OK"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
