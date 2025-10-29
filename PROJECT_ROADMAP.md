# LexiClass Project Roadmap

**Date:** October 30, 2025
**Status:** Planning Complete, Ready for Implementation
**Goal:** Transform LexiClass into a professional, extensible, production-ready ML toolkit

---

## 📋 Executive Summary

LexiClass is a well-architected document classification toolkit (~2000 LOC) with strong fundamentals but lacking production infrastructure and modern ML alternatives. This roadmap combines two parallel improvement tracks:

1. **Infrastructure & Quality** - Testing, CI/CD, documentation, packaging
2. **ML Extensibility** - Modern alternatives for tokenization, features, and classification

**Timeline:** 3-4 months (part-time)
**Effort:** ~150-180 hours total
**Risk:** Low (backward compatible, incremental approach)

---

## 📚 Planning Documents

This roadmap references two detailed planning documents:

### 1. [IMPROVEMENT_PLAN.md](./IMPROVEMENT_PLAN.md)
Comprehensive plan for infrastructure improvements:
- Testing infrastructure (0% → 85% coverage)
- CI/CD pipeline with GitHub Actions
- Type checking with mypy
- Documentation with MkDocs
- Custom exception hierarchy
- Packaging for PyPI
- Security scanning

### 2. [ML_EXTENSIBILITY_PLAN.md](./ML_EXTENSIBILITY_PLAN.md)
Detailed plan for ML capability extensions:
- Enhanced plugin system with metadata
- Alternative tokenizers (spaCy, SentencePiece, HF)
- Alternative features (TF-IDF, FastText, Sentence-BERT)
- Alternative classifiers (XGBoost, Transformers)
- CLI and library APIs
- Performance benchmarks

---

## 🎯 Key Goals

### Infrastructure Goals
✅ **Testing:** 85%+ code coverage
✅ **Automation:** Green CI on all PRs
✅ **Documentation:** Complete API docs + guides
✅ **Distribution:** Published on PyPI
✅ **Quality:** Type-checked, linted, formatted

### ML Goals
✅ **Flexibility:** 3+ options per component
✅ **Quality:** SOTA alternatives available
✅ **Speed:** Fast alternatives available
✅ **Compatibility:** 100% backward compatible
✅ **Usability:** Simple CLI/library interface

---

## 🗓️ Integrated Timeline

### Month 1: Foundation (Weeks 1-4)

**Week 1-2: Core Infrastructure**
- [ ] Set up pytest with coverage
- [ ] Write unit tests (target: 50% coverage)
- [ ] Create GitHub Actions CI/CD
- [ ] Add pre-commit hooks (ruff, mypy)
- [ ] Configure mypy type checking
- [ ] Create custom exception hierarchy
- [ ] Clean up code smells

**Deliverable:** ✓ Green CI pipeline, 50% test coverage

**Week 3-4: Enhanced Plugin System**
- [ ] Create `plugins/base.py` with metadata system
- [ ] Implement enhanced `PluginRegistry`
- [ ] Refactor existing components as plugins:
  - [ ] `tokenization.py` → `plugins/tokenizers/icu.py`
  - [ ] `features.py` → `plugins/features/bow.py`
  - [ ] Extract SVM → `plugins/classifiers/svm.py`
- [ ] Add `ClassifierProtocol` to interfaces
- [ ] Update tests for new structure

**Deliverable:** ✓ Plugin system operational, backward compatible

---

### Month 2: Quality & High-Value ML (Weeks 5-8)

**Week 5-6: Documentation & Testing**
- [ ] Set up MkDocs with Material theme
- [ ] Write API documentation
- [ ] Create user guides:
  - [ ] Getting Started
  - [ ] Building Indexes
  - [ ] Training Models
  - [ ] Plugin Development Guide
- [ ] Add CONTRIBUTING.md
- [ ] Add CHANGELOG.md
- [ ] Increase test coverage to 70%

**Deliverable:** ✓ Documentation site, 70% coverage

**Week 7-8: High-ROI ML Additions**
- [ ] Implement TF-IDF feature extractor
- [ ] Implement XGBoost classifier
- [ ] Implement spaCy tokenizer
- [ ] Add CLI support for new plugins
- [ ] Write integration tests for combinations
- [ ] Update documentation with comparisons

**Deliverable:** ✓ 3 high-value alternatives, tested and documented

---

### Month 3: Advanced Features (Weeks 9-12)

**Week 9-10: SOTA ML Capabilities**
- [ ] Implement Sentence-BERT feature extractor
- [ ] Implement Transformer classifier
- [ ] Implement FastText embeddings
- [ ] Add model download utilities
- [ ] Create configuration presets
- [ ] Performance benchmarks

**Deliverable:** ✓ SOTA alternatives available

**Week 11-12: Additional Tokenizers & Polish**
- [ ] Implement SentencePiece tokenizer
- [ ] Implement HF tokenizer
- [ ] Add `lexiclass plugins` CLI commands
- [ ] Create plugin comparison guide
- [ ] Write migration guide
- [ ] Increase test coverage to 85%+

**Deliverable:** ✓ Full plugin ecosystem, comprehensive docs

---

### Month 4: Release Preparation (Weeks 13-16)

**Week 13-14: Packaging & Security**
- [ ] Enhance pyproject.toml for PyPI
- [ ] Add badges to README
- [ ] Set up Dependabot
- [ ] Add security scanning (Bandit, pip-audit)
- [ ] Create SECURITY.md
- [ ] Test installation on clean environments
- [ ] Prepare release notes

**Deliverable:** ✓ PyPI-ready package

**Week 15: Final Testing & Documentation**
- [ ] Run full test suite on Python 3.9-3.12
- [ ] Cross-platform testing (Linux, macOS, Windows)
- [ ] Performance regression tests
- [ ] Final documentation review
- [ ] Example notebooks
- [ ] Video tutorials (optional)

**Deliverable:** ✓ Battle-tested, well-documented

**Week 16: Release**
- [ ] Create git tag v0.2.0
- [ ] Publish to PyPI
- [ ] Publish documentation
- [ ] Create GitHub release
- [ ] Announce on relevant channels
- [ ] Monitor for issues

**Deliverable:** 🚀 v0.2.0 released!

---

## 🔧 Component Alternatives Summary

### Tokenizers

| Name | Priority | Status | Dependencies | Best For |
|------|----------|--------|--------------|----------|
| **ICU** | ✅ Current | ✓ | PyICU | Locale-aware, production |
| **spaCy** | 🔥 High | ⏳ | spacy | Modern, multilingual |
| **SentencePiece** | 🔥 High | ⏳ | sentencepiece | Subword, neural ML |
| **HF Tokenizers** | 🟡 Medium | ⏳ | tokenizers | Transformers |
| **NLTK** | 🟢 Low | ⏳ | nltk | Educational, research |

### Feature Extractors

| Name | Priority | Status | Dependencies | Best For |
|------|----------|--------|--------------|----------|
| **BoW** | ✅ Current | ✓ | gensim | Baseline, fast |
| **TF-IDF** | 🔥 High | ⏳ | gensim | Better than BoW |
| **FastText** | 🔥 High | ⏳ | gensim | OOV handling |
| **Sentence-BERT** | 🔥 High | ⏳ | sentence-transformers | SOTA quality |
| **Doc2Vec** | 🟡 Medium | ⏳ | gensim | Mid-tier |

### Classifiers

| Name | Priority | Status | Dependencies | Best For |
|------|----------|--------|--------------|----------|
| **Linear SVM** | ✅ Current | ✓ | scikit-learn | Baseline, interpretable |
| **XGBoost** | 🔥 High | ⏳ | xgboost | Best traditional ML |
| **Transformer** | 🔥 High | ⏳ | transformers, torch | SOTA results |
| **Logistic Reg** | 🟡 Medium | ⏳ | scikit-learn | Fast, simple |
| **LightGBM** | 🟢 Low | ⏳ | lightgbm | Alternative to XGBoost |

**Legend:**
- 🔥 High Priority: Implement early
- 🟡 Medium Priority: Implement mid-term
- 🟢 Low Priority: Implement if time permits
- ✅ Current: Already implemented
- ⏳ Planned: To be implemented

---

## 📊 Success Metrics

### Code Quality Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Test Coverage | 0% | 85%+ | 🔴 Not started |
| Type Coverage | Unknown | 90%+ | 🔴 Not started |
| Documentation | 40% | 90%+ | 🟡 Partial |
| CI Pass Rate | N/A | 95%+ | 🔴 Not started |
| Linter Score | Unknown | 9.0+ | 🔴 Not started |

### ML Capability Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Tokenizer Options | 1 | 3+ | 🟡 1/3 |
| Feature Options | 1 | 4+ | 🟡 1/4 |
| Classifier Options | 1 | 3+ | 🟡 1/3 |
| Plugin Metadata | No | Yes | 🔴 Not started |
| Performance Benchmarks | No | Yes | 🔴 Not started |

### Community Metrics

| Metric | Current | 6-Month Target |
|--------|---------|----------------|
| PyPI Downloads | 0 | 500+/month |
| GitHub Stars | Current | +100 |
| Contributors | 1 | 5+ |
| Documentation Views | 0 | 1000+/month |
| Issues Resolved | N/A | <48h avg |

---

## 🚀 Quick Start Guide (Post-Implementation)

### For Users

```bash
# Install with all optional features
pip install lexiclass[all]

# Or install specific features
pip install lexiclass[xgboost,spacy]

# List available plugins
lexiclass plugins list

# Use preset configurations
lexiclass build-index ./texts ./index --preset balanced
lexiclass train ./index ./labels.tsv ./model.pkl

# Or customize
lexiclass build-index ./texts ./index \
  --tokenizer spacy \
  --features tfidf \
  --features-params normalize=true

lexiclass train ./index ./labels.tsv ./model.pkl \
  --classifier xgboost \
  --classifier-params n_estimators=200,use_gpu=true
```

### For Developers

```python
from lexiclass.plugins import registry

# See what's available
print(registry.list_plugins(plugin_type="feature_extractor"))

# Use any combination
tokenizer = registry.create("spacy", model_name="en_core_web_sm")
features = registry.create("sbert", model_name="all-mpnet-base-v2")
classifier = registry.create("xgboost", n_estimators=200)

# Or use configuration files
from lexiclass.config import load_preset
config = load_preset("balanced")
```

---

## 🛠️ Development Workflow

### Setting Up Development Environment

```bash
# Clone and setup
git clone https://github.com/arafat/lexiclass.git
cd lexiclass
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with all dev dependencies
pip install -e ".[dev,test,docs,all]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linters
ruff check src/
mypy src/lexiclass

# Build docs
cd docs && mkdocs serve
```

### Making Changes

1. Create feature branch: `git checkout -b feature/add-xgboost`
2. Make changes with tests
3. Run pre-commit: `pre-commit run --all-files`
4. Run tests: `pytest`
5. Update CHANGELOG.md
6. Create pull request

---

## 📦 Dependency Strategy

### Core Dependencies (Always Installed)
```
numpy>=1.22,<2.0
scipy>=1.8,<2.0
scikit-learn>=1.1,<2.0
gensim>=4.3,<5.0
beautifulsoup4>=4.12
typer[all]>=0.9
psutil>=5.9
```

### Optional Dependencies (Install as Needed)

**Tokenizers:**
```bash
pip install lexiclass[spacy]      # spaCy tokenizer
pip install lexiclass[sentencepiece]  # SentencePiece
```

**Features:**
```bash
pip install lexiclass[sbert]      # Sentence-BERT
pip install lexiclass[fasttext]   # FastText (included in gensim)
```

**Classifiers:**
```bash
pip install lexiclass[xgboost]    # XGBoost
pip install lexiclass[transformers]  # BERT, RoBERTa, etc.
```

**All Plugins:**
```bash
pip install lexiclass[all]        # Everything
```

**Development:**
```bash
pip install lexiclass[dev,test,docs]
```

---

## 🎓 Learning Path for Contributors

### Phase 1: Understanding (Week 1)
- Read README.md and CLAUDE.md
- Read IMPROVEMENT_PLAN.md
- Read ML_EXTENSIBILITY_PLAN.md
- Explore existing codebase
- Run examples

### Phase 2: Setup (Week 1)
- Set up development environment
- Run tests locally
- Build documentation locally
- Make small PR (typo fix, docstring)

### Phase 3: Contributing (Ongoing)
- Pick issue from project board
- Follow contribution guidelines
- Write tests for changes
- Update documentation
- Submit PR for review

---

## 🔐 Security Considerations

### Current State
- No security scanning
- No vulnerability monitoring
- No security policy

### Planned Improvements
- [ ] Add Bandit for security linting
- [ ] Add pip-audit for dependency vulnerabilities
- [ ] Create SECURITY.md with reporting process
- [ ] Enable Dependabot security alerts
- [ ] Add security scanning to CI
- [ ] Regular dependency updates

---

## 📈 Performance Targets

### Build Index Performance

| Corpus Size | Current (BoW+ICU) | Target (TF-IDF+spaCy) | Target (SBERT) |
|-------------|-------------------|----------------------|----------------|
| 10K docs | ~2 min | ~2 min | ~10 min |
| 100K docs | ~20 min | ~20 min | ~100 min |
| 1M docs | ~3.5 hours | ~3.5 hours | Not recommended |

### Training Performance

| Corpus Size | SVM | XGBoost | Transformer |
|-------------|-----|---------|-------------|
| 10K docs | ~10 sec | ~30 sec | ~30 min |
| 100K docs | ~2 min | ~5 min | ~5 hours |
| 1M docs | ~30 min | ~1 hour | Not feasible |

### Prediction Performance

| Corpus Size | SVM | XGBoost | Transformer |
|-------------|-----|---------|-------------|
| 1K docs | <1 sec | <1 sec | ~10 sec |
| 10K docs | ~5 sec | ~5 sec | ~100 sec |
| 100K docs | ~50 sec | ~50 sec | ~15 min |

**Note:** Times are approximate, vary by hardware and configuration

---

## 🎯 Quality Targets

### Accuracy Improvements (Expected)

**AG News Dataset (4-class classification):**
- Baseline (BoW + SVM): ~88-90%
- TF-IDF + SVM: ~90-92%
- TF-IDF + XGBoost: ~92-94%
- Sentence-BERT + XGBoost: ~94-96%
- Fine-tuned Transformer: ~95-97%

**IMDB Sentiment (Binary):**
- Baseline (BoW + SVM): ~86-88%
- TF-IDF + SVM: ~88-90%
- TF-IDF + XGBoost: ~90-92%
- Sentence-BERT + XGBoost: ~92-94%
- Fine-tuned Transformer: ~94-96%

---

## 🔄 Backward Compatibility Strategy

### Version Plan

**v0.1.x (Current):**
- Current implementation
- No plugin system

**v0.2.x (Next Release):**
- Add plugin system
- Keep old imports working
- Add deprecation warnings
- Add all new plugins
- 100% backward compatible

**v0.3.x:**
- Encourage plugin usage
- Louder deprecation warnings
- New features plugin-only

**v1.0.0 (Future):**
- Clean plugin-only API
- Remove deprecated imports
- Stable, production-ready

### Migration Support

**Documentation:**
- Migration guide for each version
- Side-by-side code examples
- Automated migration scripts (optional)

**Tooling:**
- `lexiclass migrate` command (future)
- Linter warnings for deprecated usage
- Automatic code modernization suggestions

---

## 📢 Communication Plan

### Documentation
- [ ] Update README.md with new features
- [ ] Add plugin comparison tables
- [ ] Create video tutorials
- [ ] Write blog posts for each milestone

### Community
- [ ] Create discussions forum
- [ ] Regular progress updates
- [ ] Monthly community calls (if interest)
- [ ] Contributor recognition

### Release Announcements
- [ ] GitHub releases with detailed notes
- [ ] PyPI release page
- [ ] Reddit (r/MachineLearning, r/Python)
- [ ] Twitter/X (optional)
- [ ] Relevant newsletters

---

## 🎬 Next Steps

### This Week
1. ✅ Review planning documents
2. ⏳ Approve roadmap
3. ⏳ Create GitHub project board
4. ⏳ Set up development branch
5. ⏳ Create initial issues

### Next Week (Start Implementation)
1. Set up pytest infrastructure
2. Write first unit tests
3. Create GitHub Actions workflow
4. Add pre-commit hooks
5. Configure mypy

### After That
Follow the monthly timeline outlined above, tracking progress on GitHub project board.

---

## 📋 Decision Log

### Key Decisions Made

**2025-10-30: Plugin System Architecture**
- Decision: Use Protocol-based plugins with metadata
- Rationale: Maintains flexibility while adding discoverability
- Alternative considered: Abstract base classes (too rigid)

**2025-10-30: Dependency Strategy**
- Decision: All ML alternatives as optional dependencies
- Rationale: Avoid bloat, users install only what they need
- Alternative considered: Separate packages (too fragmented)

**2025-10-30: Testing Framework**
- Decision: pytest with coverage
- Rationale: Industry standard, great ecosystem
- Alternative considered: unittest (less features)

**2025-10-30: Documentation Tool**
- Decision: MkDocs with Material theme
- Rationale: Beautiful, modern, easy to use
- Alternative considered: Sphinx (more complex)

---

## 🤝 Contribution Opportunities

### Good First Issues (After Setup)
- Add docstrings to undocumented functions
- Fix typos in documentation
- Add type hints where missing
- Write tests for untested functions
- Add examples to documentation

### Intermediate Issues
- Implement new tokenizer plugin
- Add configuration preset
- Write integration test
- Create tutorial notebook
- Performance optimization

### Advanced Issues
- Implement new feature extractor
- Implement new classifier
- Add AutoML capabilities
- Distributed processing support
- REST API wrapper

---

## 📚 References

### LexiClass Documentation
- [IMPROVEMENT_PLAN.md](./IMPROVEMENT_PLAN.md) - Infrastructure improvements
- [ML_EXTENSIBILITY_PLAN.md](./ML_EXTENSIBILITY_PLAN.md) - ML extensions
- [CLAUDE.md](./CLAUDE.md) - Architecture guide
- [README.md](./README.md) - User guide
- [dataset_preparation.md](./dataset_preparation.md) - Dataset guide

### External Resources
- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
- [Sentence-Transformers](https://www.sbert.net/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 🏆 Success Definition

**LexiClass v1.0 will be successful when:**

✅ **Professional Quality**
- 85%+ test coverage with comprehensive test suite
- Type-checked, linted, and formatted codebase
- Green CI on all commits
- Published on PyPI with semantic versioning

✅ **Well Documented**
- Complete API documentation
- User guides and tutorials
- Plugin development guide
- Video tutorials and examples

✅ **Extensible & Flexible**
- Multiple options for each component
- Easy plugin development
- Clean, protocol-based interfaces
- Configuration presets for common use cases

✅ **Production Ready**
- Battle-tested on real datasets
- Performance benchmarks published
- Security scanned and monitored
- Active maintenance and support

✅ **Community Driven**
- 5+ active contributors
- Regular releases (monthly/quarterly)
- Responsive to issues (<48h)
- Growing user base

---

## 🔮 Future Vision (Beyond v1.0)

### v1.x - Stability & Growth
- Bug fixes and optimizations
- Additional plugins based on demand
- Performance improvements
- Community plugin contributions

### v2.x - Advanced Features
- AutoML capabilities (automatic plugin selection)
- Hyperparameter tuning framework
- A/B testing for model comparison
- Model versioning and registry
- Experiment tracking integration

### v3.x - Scale & Distribution
- Distributed processing (Dask, Ray)
- Streaming data support
- Online learning capabilities
- Model serving (REST API, gRPC)
- Kubernetes deployment support

### Long-term Vision
- Industry-standard document classification toolkit
- Rich ecosystem of community plugins
- Integration with major ML platforms
- Research-to-production pipeline
- Active conference presence

---

## 📞 Contact & Support

**Project Lead:** Arafat Hasan

**Resources:**
- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: Questions, ideas, showcase
- Documentation: [ReadTheDocs/GitHub Pages]
- Email: [TBD]

**Contributing:**
- Read [CONTRIBUTING.md](./CONTRIBUTING.md) (to be created)
- Join discussions
- Pick an issue from project board
- Submit pull request

---

## 📊 Project Status Dashboard

**Last Updated:** 2025-10-30

### Overall Progress: 0% Complete

| Phase | Status | Progress | Target Date |
|-------|--------|----------|-------------|
| Planning | ✅ Complete | 100% | 2025-10-30 |
| Foundation | 🔴 Not Started | 0% | 2025-11-27 |
| Quality & ML | 🔴 Not Started | 0% | 2025-12-25 |
| Advanced | 🔴 Not Started | 0% | 2026-01-22 |
| Release Prep | 🔴 Not Started | 0% | 2026-02-19 |

### Component Status

**Infrastructure:**
- Testing: 🔴 0% coverage
- CI/CD: 🔴 Not configured
- Type Checking: 🔴 Not configured
- Documentation: 🟡 Partial
- Packaging: 🟡 Basic

**ML Plugins:**
- Tokenizers: 🟡 1/5 implemented
- Features: 🟡 1/5 implemented
- Classifiers: 🟡 1/4 implemented
- Plugin System: 🔴 Not started

---

## ✅ Conclusion

This roadmap provides a clear, structured path to transform LexiClass from a functional project into a professional, production-ready ML toolkit with modern capabilities.

**Key Strengths:**
- ✨ Comprehensive, detailed planning
- 📈 Clear success metrics
- 🔄 Backward compatible approach
- 🎯 Prioritized by ROI
- 📚 Well-documented strategy

**Next Action:** Review and approve roadmap, then begin Phase 1 (Foundation)

**Timeline:** 3-4 months to v0.2.0, 6-8 months to v1.0.0

**Let's build something great! 🚀**

---

*This roadmap is a living document. Update as priorities shift or new information emerges.*
