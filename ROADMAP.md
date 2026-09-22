# Starman Engine — Roadmap de Produto e Implementação

## Visão

Construir uma engine 3D desktop, editor-first, capaz de criar e distribuir jogos comerciais pequenos e médios em Windows, Linux e macOS.

A referência de qualidade são engines como Godot e Unreal, não a reprodução imediata de toda a abrangência delas. A Starman deve primeiro ser coerente, confiável e agradável de usar; depois ampliar sua sofisticação.

## Decisões de produto

- O primeiro produto de prova será um pequeno jogo 3D de exploração/ação com uma fase completa.
- Gameplay será escrito em Rust ou Lua.
- A engine carregará projetos e plugins dinamicamente.
- Lua terá hot reload cedo. Plugins Rust ganharão hot reload por uma ABI estável, com estado explicitamente serializável.
- O editor será o fluxo principal de autoria.
- O renderer começará em Forward+ e será organizado por um render graph modular.
- A meta gráfica inicial é PBR moderno: iluminação, sombras, HDR, pós-processamento, materiais, animação e partículas.
- Cenas aninhadas serão o modelo de composição; um prefab será uma cena instanciável com overrides explícitos.
- Formatos persistidos terão IDs estáveis, versão declarada e migrações automáticas.
- `wgpu`, `bevy_ecs`, Rapier e Kira continuam como dependências especializadas.
- Networking faz parte do produto, mas entra depois da vertical slice single-player.
- A Starman será preparada como produto público e open source, enquanto os primeiros marcos são validados pelo jogo de referência.

## Definição de pronto

Uma capacidade só está pronta quando:

1. funciona no runtime e no editor;
2. possui mensagens de erro acionáveis e não corrompe o projeto em caso de falha;
3. tem testes no nível apropriado e passa na matriz Windows, Linux e macOS;
4. possui documentação mínima para o usuário e para quem mantém o subsistema;
5. foi usada em uma fatia real do jogo de referência;
6. tem desempenho medido quando está em caminho crítico;
7. preserva ou migra dados produzidos por versões anteriores.

## Estado atual

Com o M1 fechado, a base contém:

- workspace Rust modular com crates de core, render, física, áudio, input, assets, project, reflexão, editor e runtime compartilhado (`engine-runner`);
- ECS e schedules multithreaded baseados em `bevy_ecs`;
- transform hierarchy, câmeras, janela, fixed update e frame statistics;
- reflexão de componentes e inspector orientado por metadados;
- **modelo de projeto**: manifesto versionado (`project.ron`), separação entre `assets/`, cache importado, diagnósticos e build (`crates/engine-project`), com `Project::open/validate/create` e um projeto de referência real (`examples/reference-project`);
- **identidade persistente**: UUIDs estáveis para projetos, entidades de cena e source assets (`ProjectId`, `EntityId`, `SourceAssetId`), com sub-assets reais (`SubAssetId`) para meshes multi-recurso;
- **asset database**: metadados de importação versionados (`.meta.ron`), hash de conteúdo, cache content-addressed e grafo de dependências (`crates/engine-assets/src/import`);
- **referências tipadas**: `MeshRenderer`/`Sprite` persistem por ID quando um `AssetDatabase` está anexado, com resolução por caminho legado preservada para compatibilidade (ver `docs/asset-pipeline.md`);
- **file watching** com debounce, jobs assíncronos, cancelamento por geração e atualização atômica;
- **CLI** (`starman-cli`, binário `starman`) com `new`, `validate`, `import`, `run` e `test`;
- cenas RON versionadas (v1→v2) com migração automática e testes golden-file;
- renderer `wgpu` com integração de mesh, material, textura, câmera e viewport;
- física Rapier 2D/3D, áudio Kira e input de teclado, mouse e gamepad;
- editor `egui` com docking, seleção, gizmos, undo/redo, asset browser, play mode em processo separado, e abertura de projeto via manifesto;
- hot reload de textura, mesh e material (não mais só textura) e um sandbox executável, ambos project-aware.

Os maiores gaps estruturais que restam são:

- não existe ainda um contrato de plugin/ABI estável (M3);
- cenas ainda não são componíveis (sem cenas aninhadas nem prefabs — M2);
- o renderer precisa ser separado em extração, preparação, fila e execução antes de crescer (M4);
- a garantia de referência tipada por ID só vale para cenas já salvas depois da migração — o formato legado por caminho continua lido, mas não é reescrito sozinho por uma leitura (ver "migração ao salvar" em `docs/asset-pipeline.md`);
- não há ainda uma vertical slice distribuível que prove o workflow completo.

## Ordem de dependências

```text
contratos + observabilidade + CI multiplataforma
  └─ identidade persistente + project model + asset database
      ├─ cenas aninhadas/prefabs + undo/redo transacional
      ├─ ABI de plugins + Lua + hot reload
      └─ render graph + lifecycle de recursos GPU
          └─ PBR + animação + VFX + UI
              └─ vertical slice single-player distribuível
                  ├─ ferramentas avançadas e otimização
                  └─ networking
```

Essa ordem é deliberada. Features visuais construídas antes dos contratos de dados, assets e plugins aumentariam muito o retrabalho.

## Marcos

Os marcos são gates de capacidade, não datas. Um marco termina quando seus critérios de saída são atendidos.

### M0 — Baseline confiável

**Objetivo:** tornar o estado atual reproduzível e mensurável nas três plataformas.

Entregas:

- CI para Windows, Linux e macOS com format, clippy, testes e builds de editor/runner;
- smoke test gráfico por backend disponível e testes headless para o restante;
- logging estruturado, categorias, console do editor, panic/crash reports e GPU error scopes;
- suite de benchmarks versionada para frame, transforms, serialização e assets;
- ADRs para project model, IDs persistentes, asset pipeline, render architecture e plugin ABI;
- política de compatibilidade dos formatos e matriz explícita de suporte por GPU/OS;
- exemplo mínimo que abre, executa e encerra sem vazamentos ou erros de validação.

**Gate:** qualquer clone limpo consegue verificar a engine; falhas dizem o que ocorreu e onde.

### M1 — Project model e identidade persistente

**Objetivo:** transformar o workspace numa engine que abre projetos reais.

Entregas:

- manifesto de projeto com versão, nome, entry scene, plugins, configurações e targets;
- separação clara entre instalação da engine, projeto, source assets, imported assets, cache e build;
- UUIDs estáveis para assets, entidades e sub-assets;
- referências tipadas e serializáveis sem persistir IDs efêmeros do ECS;
- asset database com metadados de importação, hash de conteúdo e grafo de dependências;
- file watching com debounce, jobs assíncronos, cancelamento e atualização atômica;
- CLI inicial: criar, validar, importar, executar e testar projeto;
- migrações de projeto/cena testadas com golden files.

**Gate:** mover ou renomear um asset não quebra referências; um projeto sobrevive a reload e migração sem perda silenciosa.

### M2 — Cenas aninhadas e prefabs

**Objetivo:** estabelecer o modelo definitivo de autoria do mundo.

Entregas:

- `SceneAsset`, instâncias de cena e resolução de referências por UUID;
- cenas aninhadas com detecção de ciclos;
- overrides por entidade/componente/campo, com aplicar, reverter e promover;
- entidades adicionadas/removidas localmente em instâncias;
- diff estrutural legível e determinístico;
- edição isolada de prefab e breadcrumb de contexto;
- clipboard, duplicação, multi-selection e undo/redo transacional;
- autosave, recovery e escrita crash-safe.

**Gate:** uma fase do jogo de referência é composta de cenas reutilizáveis e continua íntegra após renomeações, reimports e migrações.

### M3 — Runtime extensível: Rust, Lua e plugins

**Objetivo:** fazer projeto e engine evoluírem de forma independente.

Entregas:

- ABI C estreita e versionada entre host e plugins; nenhum tipo Rust instável atravessa a fronteira;
- SDK de plugin com negociação de versão, capability flags e lifecycle explícito;
- registro dinâmico de componentes, systems, assets, importers e editor extensions;
- handles opacos e APIs orientadas a comandos/queries;
- isolamento de panic/falha e unload ordenado;
- shadow-copy de bibliotecas no Windows e estratégia equivalente nos demais sistemas;
- snapshot/migração/restauração explícita do estado em reload de plugin Rust;
- runtime Lua com bindings gerados a partir da reflexão, módulos, coroutines e debugger básico;
- hot reload Lua preservando estado declarado e reportando erros sem derrubar o editor;
- permissões claras para filesystem/process/network em scripts e plugins.

**Gate:** alterar gameplay Lua recarrega durante play; recompilar um plugin Rust recarrega com estado suportado ou apresenta um diagnóstico seguro e reversível.

### M4 — Renderer escalável

**Objetivo:** criar a arquitetura que sustentará as features gráficas seguintes.

Entregas:

- pipeline `extract → prepare → queue → render`, desacoplado do world de gameplay;
- render graph declarativo com recursos, passes, dependências e inspeção no editor;
- gestão geracional de recursos GPU, upload assíncrono e deferred destruction;
- shader library, includes, variantes, reflection, cache e mensagens de compilação mapeadas ao source;
- feature/capability tiers coerentes entre Vulkan, DirectX 12 e Metal via `wgpu`;
- Forward+ com clustered light culling;
- frustum culling, instancing e batching medidos;
- picking por GPU e overlays próprios do editor;
- captura de frame e debug views para buffers, normals, clusters, shadows e overdraw.

**Gate:** o renderer suporta cenas grandes de teste sem acoplamento ao editor e sem erros de validação nos três sistemas.

### M5 — PBR e apresentação visual

**Objetivo:** atingir uma base visual moderna e previsível.

Entregas:

- materiais metallic/roughness, normal, occlusion, emissive e alpha modes;
- color management linear, HDR, exposure, tone mapping e output sRGB correto;
- directional, point e spot lights;
- sombras direcionais em cascata e sombras locais com budgets configuráveis;
- image-based lighting, skybox e reflection probes;
- MSAA/TAA conforme capabilities, bloom, SSAO e fog;
- LOD, mesh bounds e importação robusta de glTF;
- material inspector e preview no editor;
- presets de qualidade e fallback visual determinístico.

**Gate:** a fase de referência alcança uma direção de arte consistente nos três backends, com budgets documentados.

### M6 — Gameplay stack e ferramentas de conteúdo

**Objetivo:** permitir construir o jogo de referência sem ferramentas externas improvisadas.

Entregas:

- animation clips, skeletons, skinning, blend trees e state machines;
- editor de animação e eventos na timeline;
- partículas GPU/CPU com authoring no editor;
- UI de jogo com layout, texto, imagens, input/foco e localization-ready strings;
- áudio espacial, buses, mixer, snapshots e zones;
- input actions, rebinding, múltiplos dispositivos e persistência;
- character controller, joints, triggers, layers e debug de física;
- navegação, navmesh, pathfinding e ferramentas mínimas de AI;
- save-game versionado separado do formato de cena.

**Gate:** o loop completo do jogo — controlar, interagir, enfrentar um desafio, usar UI, salvar e carregar — é produzido dentro do workflow oficial.

### M7 — Editor de produção

**Objetivo:** reduzir drasticamente o custo cotidiano de criar e depurar conteúdo.

Entregas:

- project manager e criação por templates;
- hierarchy/outliner robusto, busca, filtros, collections e multi-edit;
- gizmos completos, snapping, local/global/pivot modes e atalhos configuráveis;
- asset browser com thumbnails assíncronos, tags, dependências e referências reversas;
- console pesquisável, debugger Lua, profiler CPU/GPU e frame timeline;
- build settings, project settings e input settings;
- play-in-editor com pause, step, live edit e política explícita de aplicar mudanças;
- command palette e navegação por teclado;
- extensões de editor via plugins;
- acessibilidade, DPI e persistência confiável de layouts.

**Gate:** criar, depurar e empacotar a vertical slice não exige editar RON nem usar scripts ad hoc.

### M8 — Builds distribuíveis e vertical slice

**Objetivo:** provar que Starman entrega um jogo, não apenas demos técnicas.

Entregas:

- cooker incremental e perfis development/release;
- stripping de assets e plugins não usados;
- pacotes autocontidos para Windows, Linux e macOS;
- configurações, ícones, metadata, crash logs e save paths por plataforma;
- testes em hardware representativo e GPUs integradas/dedicadas;
- jogo de referência completo, com menu, fase, áudio, UI, save/load e encerramento;
- documentação “do zero ao build” validada em máquina limpa;
- análise de tamanho, tempos de carregamento, frame time e memória.

**Gate:** terceiros conseguem baixar e jogar builds das três plataformas, e outro desenvolvedor consegue reproduzi-los a partir do projeto.

### M9 — Escala, estabilidade e ecossistema

**Objetivo:** converter uma engine funcional numa plataforma sustentável.

Entregas:

- streaming de cenas/células e carregamento assíncrono com budgets;
- occlusion culling, impostors e otimizações dirigidas por profiling;
- pipeline de shader/material mais extensível;
- API docs, manual, exemplos, templates e guia de contribuição;
- versionamento semântico da SDK, deprecações e ferramenta de migração;
- testes de compatibilidade de plugins e projetos antigos;
- release automation, changelog e artefatos assinados quando aplicável;
- ao menos um projeto externo piloto além do jogo de referência.

**Gate:** atualizar a engine não exige adivinhação, e um plugin compatível não pode derrubar silenciosamente o host.

### M10 — Networking

**Objetivo:** adicionar multiplayer sobre contratos já maduros de estado e serialização.

Entregas:

- arquitetura client/server e execução headless;
- transport abstraction, conexão, autenticação de sessão e time sync;
- replication schema versionado, relevancy e interest management;
- snapshots, delta compression e bandwidth budgets;
- prediction, interpolation e reconciliation para o character controller;
- RPC/eventos com validação de autoridade;
- scene spawning e asset/plugin compatibility handshake;
- replay determinístico onde for viável e capture de sessões para diagnóstico;
- simulador de latência, perda, jitter e reorder no editor;
- segurança básica contra payloads inválidos e ações não autorizadas.

**Gate:** uma variante multiplayer pequena do jogo de referência funciona em servidor dedicado sob condições de rede simuladas e possui métricas reproduzíveis.

## Trilhas permanentes

Essas atividades não devem virar uma “fase de polimento” tardia:

- **Compatibilidade:** fixtures de versões anteriores e migração forward-only com backup.
- **Desempenho:** budgets por subsistema, benchmarks estáveis e profiling antes de otimizar.
- **Segurança:** validação de paths e formatos, limites de recursos e fronteiras explícitas para código de terceiros.
- **Observabilidade:** logs estruturados, diagnostics com contexto, métricas e ferramentas dentro do editor.
- **Documentação:** decisões arquiteturais, contratos públicos e tutorial atualizado pela vertical slice.
- **Multiplataforma:** CI contínua nos três sistemas, não uma rodada de port no final.
- **Dogfooding:** toda capacidade é exercitada pelo jogo de referência assim que aparece.

## Fora de escopo até a vertical slice

- consoles, mobile e Web;
- ray tracing em tempo real;
- substituição de ECS, `wgpu`, Rapier ou Kira por implementações próprias;
- marketplace, contas e serviços cloud da engine;
- linguagem visual de gameplay;
- world editor de escala AAA, terrain planetário ou equivalente a Nanite/Lumen;
- compatibilidade binária arbitrária com structs Rust entre versões.

Esses itens podem entrar depois, por evidência de produto, sem contaminar as fundações atuais.

## Próximo ciclo recomendado

Com M0 e M1 fechados, o próximo ciclo inicia M2 (cenas aninhadas e prefabs), nesta ordem:

1. `SceneAsset` e instâncias de cena resolvidas por `EntityId`/`SourceAssetId` (já estáveis desde o M1), sem introduzir nenhum novo identificador efêmero;
2. cenas aninhadas com detecção de ciclos;
3. overrides por entidade/componente/campo — aplicar, reverter, promover;
4. entidades adicionadas/removidas localmente em instâncias, com diff estrutural determinístico;
5. edição isolada de prefab e breadcrumb de contexto no editor;
6. clipboard, duplicação, multi-seleção e undo/redo transacional sobre esse novo modelo;
7. autosave, recovery e escrita crash-safe;
8. recompor a fase do jogo de referência (`examples/reference-project`) usando cenas reutilizáveis, provando o gate do M2 na prática.

Cada item deve ser entregue como uma fatia vertical pequena, mantendo o workspace sempre compilável e os formatos migráveis — o mesmo padrão usado para fechar o M1.

## Indicadores de progresso

Evitar medir progresso por contagem de features. Acompanhar:

- tempo de clone limpo até abrir o projeto de referência;
- tempo de import incremental e de reload Lua/Rust;
- tempo de carregamento e transição de cenas;
- frame time CPU/GPU, p95 e p99, no cenário de benchmark;
- memória de CPU/GPU e tamanho do build;
- falhas de CI por plataforma e bugs de compatibilidade de dados;
- quantidade de passos externos/ad hoc necessários para produzir um build;
- percentual da vertical slice criado pelo workflow documentado.

O norte é simples: em cada marco, a Starman deve ficar mais capaz de entregar um jogo real e menos dependente de conhecimento implícito do autor.
