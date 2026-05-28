/** App root — React Router + Ant Design ConfigProvider. */

import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConfigProvider, Result, Button } from 'antd';
import { DashboardLayout } from './components/Layout';
import { TaskListPage } from './pages/TaskListPage';
import { TaskDetailPage } from './pages/TaskDetailPage';
import { CreateTaskPage } from './pages/CreateTaskPage';

// ErrorBoundary: catches render errors to prevent total blank page
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[ErrorBoundary]', error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 48, textAlign: 'center' }}>
          <Result
            status="error"
            title="Something went wrong"
            subTitle={this.state.error?.message || 'Unknown rendering error'}
            extra={
              <Button type="primary" onClick={() => { this.setState({ hasError: false, error: null }); window.location.reload(); }}>
                Reload Page
              </Button>
            }
          />
        </div>
      );
    }
    return this.props.children;
  }
}

const App: React.FC = () => {
  return (
    <ErrorBoundary>
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1677ff',
        },
      }}
    >
      <BrowserRouter>
        <Routes>
          <Route element={<DashboardLayout />}>
            <Route path="/" element={<TaskListPage />} />
            <Route path="/tasks/:taskId" element={<TaskDetailPage />} />
            <Route path="/create" element={<CreateTaskPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ConfigProvider>
    </ErrorBoundary>
  );
};

export default App;
